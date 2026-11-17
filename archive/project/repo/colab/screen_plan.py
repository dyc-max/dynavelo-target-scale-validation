"""Pure frozen design, baseline-frame and independent-review gates.

No model imports/execution or outcome access. A plan is not scientific approval.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from human_return_common import safe_relative, safe_target, sha256

SHA = re.compile('[0-9a-f]{64}')
STAGES = {'baseline', 'matched-baseline', 'screen', 'grn', 'full-cell-review'}
BRANCHES = {'erythroid':['HSC','MEP','Erythrocyte'],
            'megakaryocytic':['HSC','MEP','Prog MK','Megakaryocyte','Platelet'],
            'granulocytic':['HSC','GMP','Granulocyte'], 'dendritic':['HSC','Prog DC','DC']}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def seal(value, key):
    return {**value, key: identity(value)}

def check_seal(value, key):
    body = {name:item for name,item in value.items() if name != key}
    if not SHA.fullmatch(value.get(key,'')) or identity(body) != value[key]:
        raise ValueError('Frozen identity changed: ' + key)
    return body

def names(values, label):
    if not values or any(not isinstance(value,str) or not value.strip() for value in values) or len(set(values)) != len(values):
        raise ValueError('Invalid/nonunique ' + label)
    return values

def artifact(value):
    safe_relative(value['relative_path'])
    if type(value.get('bytes')) is not int or value['bytes'] < 1 or not SHA.fullmatch(value.get('sha256','')):
        raise ValueError('Invalid artifact byte/hash identity')
    return value

def input_binding(spec):
    axes={'cells':identity([item['cell_id'] for item in spec['cells']]),
          'rna':identity([item['feature_id'] for item in spec['genes']]),
          'rna_stable_ids':identity([item['stable_id'] for item in spec['genes']]),
          'motifs':identity([item['feature_id'] for item in spec['motifs']]),
          'motif_mapping':identity([item['tf_rna_feature_id'] for item in spec['motifs']]),
          'labels_and_splits':identity(spec['cells'])}
    return {'axes':axes,'sha256':identity({'artifacts':spec['input_artifacts'],'axes':axes,'minima':spec['global_minima']})}

def sample_cells(cells, eligible_ids, seed):
    """Frozen default_rng algorithm, sorted labels/IDs and original frame weights."""
    eligible_ids = set(eligible_ids)
    frame = [cell for cell in cells if cell['cell_id'] in eligible_ids]
    if len(frame) != len(eligible_ids) or not frame:
        raise ValueError('Unknown or empty sampling frame')
    rng = np.random.default_rng(seed)
    sampled, strata = [], []
    for label in sorted({cell['label'] for cell in frame}):
        ids = sorted(cell['cell_id'] for cell in frame if cell['label'] == label)
        n = min(64,len(ids))
        chosen = sorted(rng.choice(np.asarray(ids,dtype=object), size=n, replace=False).tolist())
        strata.append({'label':label,'population_n':len(ids),'sample_n':n,
                       'inclusion_probability':n/len(ids),'population_weight':len(ids)/len(frame)})
        sampled.extend({'cell_id':cell,'label':label,'inclusion_probability':n/len(ids),
                        'population_weight':len(ids)/len(frame),'cell_weight':len(ids)/len(frame)/n} for cell in chosen)
    return {'sampling_seed':seed,'n_population':len(frame),'n_sample':len(sampled),'strata':strata,
            'cells':sampled,'cell_axis_sha256':identity([cell['cell_id'] for cell in sampled])}

def build_plan(spec):
    spec = json.loads(canonical(spec))  # Own immutable JSON-compatible content.
    if (spec.get('schema') != 'dynavelo_screen_spec_v1' or not isinstance(spec.get('prepared_by'),str)
            or not spec['prepared_by'].strip() or not isinstance(spec.get('population_id'),str) or not spec['population_id'].strip()):
        raise ValueError('Explicit specification identity and preparer required')
    if spec.get('purpose') not in {'human_exploratory_input_sensitivity','engineering_fixture'}:
        raise ValueError('Unsupported purpose')
    cells, genes, motifs, models = spec['cells'],spec['genes'],spec['motifs'],spec['models']
    cell_ids = names([cell['cell_id'] for cell in cells],'cell axis')
    gene_ids = names([gene['feature_id'] for gene in genes],'RNA feature axis')
    stable_ids = names([gene['stable_id'] for gene in genes],'RNA stable IDs')
    motif_ids = names([motif['feature_id'] for motif in motifs],'motif feature axis')
    motif_gene_ids = names([motif['tf_rna_feature_id'] for motif in motifs],'selected motif-to-RNA mapping')
    if len(cells) < 2 or not set(motif_gene_ids) <= set(gene_ids):
        raise ValueError('Invalid population/motif mapping')
    for cell in cells:
        if not isinstance(cell.get('label'),str) or not cell['label'] or cell.get('split') not in {'train','validation'}:
            raise ValueError('Missing author-state label or exact split')
    for gene in genes:
        if (type(gene.get('eligible_tf')) is not bool or type(gene.get('curated_tf')) is not bool
                or gene['eligible_tf'] and not gene['curated_tf']
                or not np.isfinite(gene['training_variance']) or gene['training_variance'] < 0):
            raise ValueError('Invalid TF eligibility/training variance')
    for motif in motifs:
        if not motif.get('source_motif_id') or not motif.get('ambiguity_annotation'):
            raise ValueError('Motif provenance and ambiguity statement required')
    axes = {'cells':identity(cell_ids),'rna':identity(gene_ids),'rna_stable_ids':identity(stable_ids),
            'motifs':identity(motif_ids),'motif_mapping':identity(motif_gene_ids),
            'labels_and_splits':identity(cells)}
    inputs = spec['input_artifacts']
    for required in ('rna','atac','tf_catalog','metadata','minima_audit','validation_protocol','validation_gates','execution_design','streaming_core','plan_module','registry_module','shortlist_module'):
        artifact(inputs[required])
    minima = spec['global_minima']
    if minima.get('scope') != 'full_frozen_population' or minima.get('cell_axis_sha256') != axes['cells']:
        raise ValueError('Minima must bind the full frozen population')
    for key,width in [('rna',len(genes)),('motif',len(motifs))]:
        vector = np.asarray(minima[key],dtype=np.float32)
        if vector.shape != (width,) or not np.isfinite(vector).all():
            raise ValueError('Invalid full-population minima')
        if key == 'rna' and (vector < 0).any():
            raise ValueError('Negative RNA floor')
    if not all(SHA.fullmatch(minima.get(key,'')) for key in ('rna_matrix_sha256','motif_matrix_sha256')):
        raise ValueError('Minima need model-input matrix hashes')
    input_identity = identity({'artifacts':inputs,'axes':axes,'minima':minima})
    if input_binding(spec)!={'axes':axes,'sha256':input_identity}:
        raise AssertionError('Internal input identity implementations differ')
    if sorted(model['seed'] for model in models) != [0,1,2] or any(type(model['seed']) is not int for model in models):
        raise ValueError('Exactly three independently identified seeds 0,1,2 required')
    names([model['run_id'] for model in models],'model run IDs')
    checkpoint_hashes = []
    for model in models:
        for key in ('checkpoint','config','metrics'):
            artifact(model[key])
        if (model.get('checkpoint_role') != 'canonical_best' or model.get('input_identity_sha256') != input_identity
                or model.get('benchmark_only') is not False):
            raise ValueError('Model needs same-input canonical best checkpoint, not benchmark')
        checkpoint_hashes.append(model['checkpoint']['sha256'])
    if len(set(checkpoint_hashes)) != 3:
        raise ValueError('Three seed checkpoints must have distinct identities')
    universe = [{'feature_id':gene['feature_id'],'stable_id':gene['stable_id'],
                 'has_selected_motif':gene['feature_id'] in motif_gene_ids,
                 'zero_training_variance':gene['training_variance']==0} for gene in genes if gene['eligible_tf']]
    if not universe or len(universe) != spec['expected_eligible_tf_count']:
        raise ValueError('Full eligible TF denominator differs from specification')
    if len({cell['label'] for cell in cells}) != spec['expected_state_count']:
        raise ValueError('State count differs from frozen metadata')
    units=spec['independent_units']
    if type(units.get('donor_mapping_verified')) is not bool:
        raise ValueError('Explicit donor-mapping status required; never infer from cells/states')
    for key in ('library_count','verified_biological_replicates'):
        if units.get(key) is not None and (type(units[key]) is not int or units[key]<1):
            raise ValueError('Independent-unit count must be known positive integer or null')
    endpoint_h = [gene['feature_id'] for gene in genes if gene['training_variance']>0 and not gene['curated_tf'] and gene['feature_id'] not in motif_gene_ids]
    if not endpoint_h:
        raise ValueError('No eligible fixed non-TF output set H')
    policy = {'sampling':{'seeds':[73011,73012],'per_state_cap':64,'algorithm':'sorted labels and IDs; numpy default_rng choice without replacement; sort selected IDs',
                          'final_frame':'common baseline-only evaluable cell set across all 3 seeds; lock before any TF effect',
                          'common_mask_algorithm':'finite positive norms, then monotone 1e-8*common-frame RMS fixed point; empty state is HOLD'},
              'primary':{'mode':'rna-only','endpoint':'signed_projection_on_matched_baseline_velocity_H_divided_by_full_frame_RMS_norm',
                         'H':endpoint_h,'stratum_weighting':'N_l/N * sample_mean_l','near_zero_threshold':1e-8},
              'secondary':{'mode':'official-joint','scope':'fixed top20 with selected motif; unmapped is not_applicable, not zero'},
              'time':{'global_rank':'stable ordinal on entire original population, separately per seed','windows':5,'perturbed_rebinning':False},
              'shortlist':{'maximum':20,'rank':'median of 3 per-seed percentiles of absolute primary E','tie_break':'stable_id lexical',
                           'per_seed_percentile_ties':'average rank; top5 membership percentile>=0.95',
                           'boundary_ties_reported':True,'replacement_after_failure':False,'all_TFs_and_both_samples_required_before_ranking':True},
              'stability':{'top_fraction':0.05,'minimum_seeds':2,'seeds_total':3},
              'grn':{'epsilon':1e-4,'input_spaces':['rna','motif'],'channel_units_not_pooled':True},
              'technical_attempt_limit':3,'scientific_or_external_validation_automatic_PASS':False,
              'independent_units':'metadata only; cells/states/seeds are not biological replicates','branches':BRANCHES}
    chunk = spec['chunk_geometry']
    if type(chunk['cells']) is not int or chunk['cells']<2 or type(chunk['targets']) is not int or chunk['targets']<1:
        raise ValueError('Invalid frozen chunks')
    result = {'schema':'dynavelo_screen_plan_v1','spec':spec,'axes_sha256':axes,'input_identity_sha256':input_identity,
              'tf_universe':universe,'policy':policy,'checkpoint_set_sha256':identity(sorted(checkpoint_hashes)),
              'proposed_samples_before_baseline':{str(seed):sample_cells(cells,cell_ids,seed) for seed in (73011,73012)},
              'final_sampling_frame_status':'PENDING_FULL_BASELINES','scientific_review_status':'SEPARATE_REVIEW_REQUIRED',
              'numpy_sampling_version':np.__version__}
    return seal(result,'plan_sha256')

def validate_plan(plan):
    check_seal(plan,'plan_sha256')
    if canonical(build_plan(plan['spec'])) != canonical(plan):
        raise ValueError('Derived plan fields do not reproduce frozen specification')
    return plan

def freeze_baseline_frame(plan, summaries):
    """Full-population baseline summaries only; never candidate-specific masks."""
    validate_plan(plan)
    cells = plan['spec']['cells']; n = len(cells)
    if sorted(summary['seed'] for summary in summaries) != [0,1,2]:
        raise ValueError('All three full baselines required before sample freeze')
    times, norms, evidence = {}, {}, []
    for summary in summaries:
        seed = summary['seed']; model = next(item for item in plan['spec']['models'] if item['seed']==seed)
        artifact(summary['artifact'])
        if (summary.get('schema') != 'dynavelo_full_baseline_summary_v1' or summary.get('plan_sha256') != plan['plan_sha256']
                or summary.get('checkpoint_sha256') != model['checkpoint']['sha256']
                or summary.get('cell_axis_sha256') != plan['axes_sha256']['cells']
                or summary.get('H') != plan['policy']['primary']['H']):
            raise ValueError('Baseline summary identity/axis/H mismatch')
        t, norm = np.asarray(summary['raw_time_mean'],dtype=float), np.asarray(summary['direction_norm_H'],dtype=float)
        if t.shape != (n,) or norm.shape != (n,) or not np.isfinite(t).all() or np.any(norm[np.isfinite(norm)]<0):
            raise ValueError('Invalid full baseline dimensions/times/norms')
        times[seed], norms[seed] = t,norm
        evidence.append({'seed':seed,'artifact':summary['artifact'],'summary_content_sha256':identity(summary)})
    mask = np.logical_and.reduce([np.isfinite(norms[seed]) & (norms[seed]>0) for seed in (0,1,2)])
    scales = {}
    for iteration in range(n+1):
        if not mask.any():
            raise ValueError('No jointly evaluable baseline cells')
        scales = {seed:float(np.sqrt(np.mean(np.square(norms[seed][mask]),dtype=np.float64))) for seed in (0,1,2)}
        if any(not np.isfinite(value) or value<=0 for value in scales.values()):
            raise ValueError('Invalid baseline scale')
        reduced = mask & np.logical_and.reduce([norms[seed]>=1e-8*scales[seed] for seed in (0,1,2)])
        if np.array_equal(mask,reduced):
            break
        mask = reduced
    else:
        raise ValueError('Baseline-only common mask failed to stabilize')
    eligible = [cell['cell_id'] for cell,keep in zip(cells,mask) if keep]
    empty = sorted({cell['label'] for cell in cells} - {cell['label'] for cell,keep in zip(cells,mask) if keep})
    windows = {}
    for seed in (0,1,2):
        ranks = np.empty(n,dtype=float)
        ranks[np.argsort(times[seed],kind='stable')] = np.arange(n)/max(n-1,1)
        windows[str(seed)] = {'raw_time_sha256':identity(times[seed].tolist()),'global_rank':ranks.tolist(),
                              'window_index':np.minimum((ranks*5).astype(int),4).tolist(),'common_frame_velocity_scale':scales[seed]}
    result = {'schema':'dynavelo_screen_frame_v1','plan_sha256':plan['plan_sha256'],
              'status':'HOLD_EMPTY_STATE_AFTER_BASELINE_FILTER' if empty else 'READY_FOR_SEPARATE_FRAME_REVIEW',
              'common_eligible_cell_ids':eligible,'common_cell_axis_sha256':identity(eligible),
              'excluded_cell_ids':[cell['cell_id'] for cell,keep in zip(cells,mask) if not keep],
              'empty_states':empty,'full_population_n':n,'baseline_evidence':evidence,'global_time_windows':windows,
              'mask_algorithm':'finite positive norm; monotone 1e-8*common-frame RMS fixed point across all three seeds',
              'mask_iterations':iteration+1,
              'samples':{name:sample_cells(cells,eligible,seed) for name,seed in [('primary',73011),('stability',73012)]},
              'scientific_scope':'Internal model input-sensitivity frame, not biological replication or external confirmation'}
    return seal(result,'frame_sha256')

def validate_frame(plan,frame):
    validate_plan(plan); check_seal(frame,'frame_sha256')
    if frame.get('schema')!='dynavelo_screen_frame_v1' or frame.get('plan_sha256')!=plan['plan_sha256']:
        raise ValueError('Frame belongs to different plan')
    eligible = frame['common_eligible_cell_ids']
    if identity(eligible)!=frame['common_cell_axis_sha256']:
        raise ValueError('Frame cell axis mismatch')
    original = [cell['cell_id'] for cell in plan['spec']['cells']]
    if len(set(eligible))!=len(eligible) or not set(eligible)<=set(original):
        raise ValueError('Unknown/duplicate frame cells')
    for name,seed in [('primary',73011),('stability',73012)]:
        if canonical(frame['samples'][name])!=canonical(sample_cells(plan['spec']['cells'],eligible,seed)):
            raise ValueError('Frozen sample/weights fail deterministic reproduction')
    for seed in ('0','1','2'):
        window = frame['global_time_windows'][seed]
        ranks = np.asarray(window['global_rank'],dtype=float)
        if ranks.shape!=(len(original),) or not np.array_equal(np.sort(ranks),np.arange(len(original))/max(len(original)-1,1)):
            raise ValueError('Global rank does not cover full original population exactly')
        if window['window_index']!=np.minimum((ranks*5).astype(int),4).tolist():
            raise ValueError('Global window assignment mismatch')
    return frame

def verify_artifact(root,record):
    artifact(record)
    path = safe_target(root,record['relative_path'])
    if not path.is_file() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
        raise ValueError('Artifact absent/changed: '+record['relative_path'])
    return path

def authorize_stage(plan, review_record, root, stage, *, trusted_reviewers=(), frame=None):
    """A separate externally trusted review is mandatory; no self-approval option."""
    validate_plan(plan)
    if stage not in STAGES or not review_record or not trusted_reviewers:
        raise PermissionError('Explicit stage and separate trusted review required')
    path = verify_artifact(root,review_record)
    if not review_record['relative_path'].startswith('results/reports/'):
        raise PermissionError('Independent review must be a report artifact')
    review = json.loads(path.read_text(encoding='utf-8'))
    if (review.get('schema')!='dynavelo_science_review_v1'
            or review.get('decision')!='APPROVED_FOR_EXPLICIT_EXPLORATORY_SCOPE'
            or review.get('plan_sha256')!=plan['plan_sha256']
            or review.get('checkpoint_set_sha256')!=plan['checkpoint_set_sha256']
            or review.get('population_id')!=plan['spec']['population_id']
            or review.get('purpose')!=plan['spec']['purpose']
            or review.get('reviewer_id') not in trusted_reviewers
            or review.get('reviewer_id')==plan['spec']['prepared_by']
            or review.get('independent_of_plan_preparer') is not True
            or review.get('model_quality_gate')!='PASS'
            or review.get('claim_scope')!='exploratory_model_input_sensitivity_only'
            or review.get('external_confirmation_or_causal_rescue_authorized') is not False
            or stage not in review.get('allowed_stages',[])
            or not set(review.get('allowed_stages',[]))<=STAGES):
        raise PermissionError('Review missing, held, untrusted, or outside exact plan/model/stage scope')
    for requirement in ('model_quality','direction','controls','resource_budget','claim_scope'):
        verify_artifact(root,review['evidence'][requirement])
    for record in plan['spec']['input_artifacts'].values():
        verify_artifact(root,record)
    for model in plan['spec']['models']:
        for key in ('checkpoint','config','metrics'):
            verify_artifact(root,model[key])
    if stage!='baseline':
        if frame is None:
            raise PermissionError('Locked shared baseline frame required')
        validate_frame(plan,frame)
        if frame['status']!='READY_FOR_SEPARATE_FRAME_REVIEW' or review.get('frame_sha256')!=frame['frame_sha256']:
            raise PermissionError('Frame is held or was not explicitly reviewed')
        summaries=[]
        for item in frame['baseline_evidence']:
            baseline_path=verify_artifact(root,item['artifact'])
            body=json.loads(baseline_path.read_text(encoding='utf-8'))
            if 'artifact' in body:raise ValueError('Baseline summary cannot contain a self-referential artifact hash')
            summaries.append({**body,'artifact':item['artifact']})
        if canonical(freeze_baseline_frame(plan,summaries))!=canonical(frame):
            raise PermissionError('Frame does not reproduce the verified full baseline summaries')
    return {'stage':stage,'review_sha256':review_record['sha256'],'plan_sha256':plan['plan_sha256'],
            'frame_sha256':frame['frame_sha256'] if frame else None,
            'status':'SCOPED_PREFLIGHT_ONLY_NOT_EXECUTED','external_confirmation_authorized':False}
