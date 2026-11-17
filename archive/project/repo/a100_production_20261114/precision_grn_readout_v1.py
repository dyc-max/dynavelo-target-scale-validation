"""Read-only complete-family FP64 GRN aggregation with immutable backed arrays.

Default metadata dry run. --execute requires all4668+84 COMMITs and parents.
No model, GPU, cloud, new external source, hypothesis reselection or patient P.
"""
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
import argparse,csv,json,math,sys,time
import numpy as np
ROOT=Path(r'D:\dyc_7\DynaVelo_Research')
CODE=ROOT/'repo/a100_production_20261114'
sys.path[:0]=[str(CODE),str(ROOT/'repo/colab')]
from recovery_io_v1 import read,checked,record,sha,new,now
from screen_plan import canonical,identity,check_seal
from precision_plan_v1 import build_books
from precision_worker_v1 import PrecisionRegistry
import screen_scope_followup_numerics_v2 as original_numerics

STAGES=('grn','full-cell-grn')
FIELDS=('sum','abs_sum','count','positive','negative','zero','nonfinite',
        'weighted_signed_sum','weighted_absolute_sum','weighted_finite_mass','weighted_nonfinite_mass')
INT_FIELDS={'count','positive','negative','zero','nonfinite'}
QUAL_STATUSES={'QUALIFIED_DESCRIPTIVE','HOLD','LOW_NUMERICAL_SIGNAL','NOT_TESTED'}

class ReadOnlyPrecisionRegistry(PrecisionRegistry):
    def __init__(self,*args,**kwargs):
        root=Path(args[0]);book=args[1]
        if not root.is_dir() or not (root/'task_book.json').is_file():raise FileNotFoundError('Returned immutable registry/book missing')
        if (root/'task_book.json').read_bytes()!=canonical(book):raise ValueError('Returned full task book differs')
        super().__init__(*args,**kwargs)
    def atomic_new(self,*args,**kwargs):raise PermissionError('Read-only precision readout')
    def commit(self,*args,**kwargs):raise PermissionError('Read-only precision readout')
    @contextmanager
    def locked(self,*args,**kwargs):raise PermissionError('Readout never reserves numerical attempts');yield

def divide_or_nan(numerator,denominator):
    result=np.full(np.asarray(numerator).shape,np.nan,dtype=np.float64)
    np.divide(numerator,denominator,out=result,where=np.asarray(denominator)>0)
    if not np.isfinite(result[np.asarray(denominator)>0]).all():raise ValueError('Derived finite-mass mean overflowed')
    return result

def parse_edge_payload(path,axes,task,*,structural_atol=1e-12):
    """Parse all stats/targets; exact zero stays zero, missing mass becomes NaN.

    atol only checks floating count/mass identities; it is no scientific cutoff.
    """
    width=len(axes['rna_output_axis'] if Path(path).name.startswith('vx') else axes['motif_output_axis'])
    groups=axes['groups'];shape=(len(groups),width)
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(FIELDS):raise ValueError('Edge statistic family differs')
        arrays={k:np.array(archive[k],copy=True) for k in FIELDS}
    for name,a in arrays.items():
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Edge shape or finite integrity differs: '+name)
        if name in INT_FIELDS:
            if a.dtype!=np.int64 or np.any(a<0):raise ValueError('Nonnegative exact int64 count required')
        elif a.dtype!=np.float64:raise ValueError('FP64 edge statistic required')
    if not np.array_equal(arrays['positive']+arrays['negative']+arrays['zero'],arrays['count']):raise ValueError('Sign counts do not sum to finite count')
    sample_n=np.array([g['sample_n'] for g in groups],dtype=np.int64)[:,None]
    if not np.array_equal(arrays['count']+arrays['nonfinite'],np.broadcast_to(sample_n,shape)):raise ValueError('Cell denominator differs')
    mass=np.array([g['sample_weight_sum'] for g in groups],dtype=np.float64)[:,None]
    if not np.allclose(arrays['weighted_finite_mass']+arrays['weighted_nonfinite_mass'],mass,rtol=structural_atol,atol=structural_atol):raise ValueError('Weighted cell denominator differs')
    if any(np.any(arrays[k]<0) for k in ('abs_sum','weighted_absolute_sum','weighted_finite_mass','weighted_nonfinite_mass')):raise ValueError('Negative absolute statistic or mass')
    if any(np.any(arrays[k][arrays['count']==0]!=0) for k in ('sum','abs_sum')):raise ValueError('Zero count carries nonzero unweighted sum')
    if any(np.any(arrays[k][arrays['weighted_finite_mass']==0]!=0) for k in ('weighted_signed_sum','weighted_absolute_sum')):raise ValueError('Zero mass carries nonzero weighted sum')
    # Successful original numerical tasks forbid nonfinite derivative entries.
    if np.any(arrays['nonfinite']) or np.any(arrays['weighted_nonfinite_mass']):raise ValueError('Complete task contains nonfinite derivative counts')
    arrays['unweighted_signed_mean']=divide_or_nan(arrays['sum'],arrays['count'])
    arrays['unweighted_absolute_mean']=divide_or_nan(arrays['abs_sum'],arrays['count'])
    arrays['weighted_signed_mean']=divide_or_nan(arrays['weighted_signed_sum'],arrays['weighted_finite_mass'])
    arrays['weighted_absolute_mean']=divide_or_nan(arrays['weighted_absolute_sum'],arrays['weighted_finite_mass'])
    return arrays

def expected_axes(original,frame,task):
    context=original['context_plan'];positions={c['cell_id']:i for i,c in enumerate(context['spec']['cells'])}
    indices=np.array([positions[c] for c in task['sample_cell_ids']],dtype=np.int64)
    if len(indices)!=task['n_cells'] or identity(task['sample_cell_ids'])!=task['sample_cell_axis_sha256']:raise ValueError('Task sample axis differs')
    weights=original_numerics.frame_weights(original,frame,task,indices)
    definitions=original_numerics.group_definitions(original,frame,task,indices)
    return {'rna_output_axis':[g['feature_id'] for g in context['spec']['genes']],
        'motif_output_axis':[m['feature_id'] for m in context['spec']['motifs']],
        'motif_output_tf_mapping':[m['tf_rna_feature_id'] for m in context['spec']['motifs']],
        'groups':[{'kind':kind,'name':name,'sample_n':int(mask.sum()),'effect_frame_domain_n':int(n),
            'sample_weight_sum':float(weights[mask].sum())} for kind,name,mask,n in definitions],
        'input_space':task['input_space'],'input_feature_id':task['feature_id'],
        'mapped_rna_feature':original_numerics.regulator_name(original,task)}

def validate_axes(axes,expected):
    for key,value in expected.items():
        if canonical(axes.get(key))!=canonical(value):raise ValueError('Edge axis/group differs: '+key)

def qualification_index(path,expected_sha,precision_plan,review_record):
    if path is None:return {},None
    if not path.is_absolute() or not path.is_relative_to(ROOT) or sha(path)!=expected_sha:raise ValueError('Qualification map path/hash differs')
    value=read(path)
    if value.get('schema')!='dynavelo_precision_grn_qualification_map_v1' or value.get('precision_plan_sha256')!=precision_plan['precision_plan_sha256']:raise ValueError('Wrong qualification map identity')
    if value.get('actual_qualification_review')!=review_record:raise ValueError('Qualification map review binding differs')
    rows={}
    for row in value['rows']:
        key=tuple(row[k] for k in ('sample_id','seed','input_space','feature_id','channel'))
        if key in rows or row['status'] not in QUAL_STATUSES or not row.get('reason'):raise ValueError('Duplicate/invalid qualification row')
        rows[key]=row
    return rows,record(ROOT,path)

def qualify(rows,task,channel):
    key=tuple(task[k] for k in ('sample_id','seed','input_space','feature_id'))+(channel,)
    return rows.get(key,{'status':'NOT_TESTED','reason':'No exact sample/seed/input/channel numerical qualification row supplied; not inferred from significance.'})

def load_definition(path,digest):
    if not path.is_absolute() or not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('Plan path/hash differs')
    plan=read(path);check_seal(plan,'precision_plan_sha256')
    for source in plan['source_records']:checked(ROOT,source)
    checked(ROOT,plan['review'])
    original,frame,shortlist=[read(checked(ROOT,plan[k])) for k in ('original_scope_plan','scope_frame','shortlist')]
    books=build_books(plan,original,frame,shortlist)
    return plan,original,frame,shortlist,books

def metadata_dry_run(path,digest):
    plan,original,frame,shortlist,books=load_definition(path,digest)
    return {'status':'READOUT_METADATA_ONLY_NO_GRN_VALUES_OPENED','precision_plan':record(ROOT,path),
        'expected_stages':{k:len(books[k]['tasks']) for k in STAGES},'baseline_family':len(books['baseline']['tasks']),
        'frozen_candidates':[r['stable_id'] for r in shortlist['selected']],
        'reference_time_windows':'Original frozen FP32 frame windows; explicitly retained reference strata',
        'patient_P_values_computed':False,'scientific_pass':False}

def iter_input_edge_blocks(index_path,input_entry,*,require_qualified=False):
    """Stream every target/group for one indexed input across all three seeds.

    Raw source hashes are rechecked. No array member/zero/negative silently drops.
    Missing qualification is NOT_TESTED; requesting a qualified-only view yields
    nothing for such a channel and does not change the complete backed index.
    """
    index=read(index_path)
    if input_entry not in index['inputs']:raise ValueError('Input entry not from exact completed index')
    by_seed={}
    for item in input_entry['seed_tasks']:
        axes=read(checked(ROOT,item['edge_axes']))
        for channel in ('vx','vy'):
            q=item['qualification'][channel]
            arrays=parse_edge_payload(checked(ROOT,item[channel+'_edges']),axes,item['task'])
            if require_qualified and q['status']!='QUALIFIED_DESCRIPTIVE':continue
            targets=axes['rna_output_axis'] if channel=='vx' else axes['motif_output_axis']
            for group_index,group in enumerate(axes['groups']):
                yield {'stage':input_entry['stage'],'input_space':input_entry['input_space'],'feature_id':input_entry['feature_id'],
                    'seed':item['task']['seed'],'channel':channel,'group':group,'target_axis':targets,
                    'values':{k:v[group_index] for k,v in arrays.items()},'qualification':q,
                    'source_commit':item['commit'],'source_edge_artifact':item[channel+'_edges']}

def describe_axis(index_path,*,stage,input_space,input_feature,channel,target_feature,group_kind,group_name):
    """Return all seed values/evidence for an explicitly requested modeled axis."""
    index=read(index_path);entry=next(r for r in index['inputs'] if (r['stage'],r['input_space'],r['feature_id'])==(stage,input_space,input_feature))
    rows=[]
    for block in iter_input_edge_blocks(index_path,entry):
        if block['channel']!=channel or (block['group']['kind'],block['group']['name'])!=(group_kind,group_name):continue
        position=block['target_axis'].index(target_feature)
        values={k:float(v[position]) if np.isfinite(v[position]) else None for k,v in block['values'].items()}
        rows.append({'seed':block['seed'],'values':values,'group':block['group'],'qualification':block['qualification'],
            'commit':block['source_commit'],'edge_artifact':block['source_edge_artifact']})
    if sorted(r['seed'] for r in rows)!=[0,1,2]:raise ValueError('Axis must retain all three seeds')
    return {'stage':stage,'input_space':input_space,'input_feature':input_feature,'channel':channel,'target_feature':target_feature,
        'group_kind':group_kind,'group_name':group_name,'all_seed_rows':rows,'interpretation':'Local finite input sensitivity; not direct binding/causality or patient replication','scientific_pass':False}

def aggregate(path,digest,output,*,qualification_map=None,qualification_sha=None):
    """Verify both full GRN families, then aggregate every input without pruning."""
    from scipy.stats import rankdata
    started=time.monotonic();plan,original,frame,shortlist,books=load_definition(path,digest)
    if {s:len(books[s]['tasks']) for s in STAGES}!={'grn':4668,'full-cell-grn':84}:raise ValueError('Full frozen GRN family differs')
    if not output.is_absolute() or not output.is_relative_to(ROOT/'results'):raise ValueError('New absolute project results output required')
    output.mkdir(parents=True,exist_ok=False)
    namespace=ROOT/plan['namespace_relative']
    # Prove all books exist before the original registry constructor can run.
    for stage in ('baseline','epsilon')+STAGES:
        if not (namespace/stage/'task_book.json').is_file() or (namespace/stage/'task_book.json').read_bytes()!=canonical(books[stage]):raise ValueError('Returned full book missing/different: '+stage)
    kwargs={'precision_plan':plan,'original_scope_plan':original,'frame':frame,'shortlist':shortlist}
    base_registry=ReadOnlyPrecisionRegistry(namespace/'baseline',books['baseline'],**kwargs)
    base_coverage=base_registry.coverage()
    if not base_coverage['all_expected_tasks_complete'] or base_coverage['expected_tasks']!=12:raise PermissionError('All12 original precision parent baselines must be complete')
    review_record=read(namespace/'actualFP64qualification_review_record.json');review=read(checked(ROOT,review_record))
    if review.get('decision')!='APPROVED_FOR_FIXED_FP64_DESCRIPTIVE_GRN' or review.get('precision_plan_sha256')!=plan['precision_plan_sha256']:raise PermissionError('Actual complete-family FP64 review absent')
    checked(ROOT,review['epsilon_comparison'])
    eps_registry=ReadOnlyPrecisionRegistry(namespace/'epsilon',books['epsilon'],**kwargs)
    eps_coverage=eps_registry.coverage()
    if not eps_coverage['all_expected_tasks_complete'] or review.get('epsilon_coverage_sha256')!=eps_coverage['coverage_sha256']:raise PermissionError('Qualification did not bind exact complete epsilon family')
    new(output/'epsilon_qualification_coverage.json',eps_coverage)
    qrows,qrecord=qualification_index(qualification_map,qualification_sha,plan,review_record)
    census=[];inputs={};coverage_records={};registries={}
    # No edge array values opened until both complete families pass original
    # COMMIT/parent validation. Failures are not redefined as a smaller family.
    for stage in STAGES:
        registry=ReadOnlyPrecisionRegistry(namespace/stage,books[stage],**kwargs);registries[stage]=registry
        coverage=registry.coverage()
        new(output/(stage+'_coverage.json'),coverage);coverage_records[stage]=record(ROOT,output/(stage+'_coverage.json'))
        if not coverage['all_expected_tasks_complete']:raise PermissionError('Entire frozen family not complete: '+stage)
        with registry.dependency_audit():
            for task in books[stage]['tasks']:
                saved=registry.resume(task['task_id']);assert saved['status']=='COMPLETE'
                directory=registry.path(saved['manifest_relative_path']).parent
                value=read(directory/'result.json')
                if value.get('review_sha256')!=review_record['sha256']:raise ValueError('Task qualification review differs')
                members={r['name']:r for r in saved['files']}
                def artifact(name):
                    r=members[name];return {'relative_path':(directory/name).relative_to(ROOT).as_posix(),'bytes':r['bytes'],'sha256':r['sha256']}
                item={'task':{k:task[k] for k in ('task_id','seed','stage','sample_id','n_cells','input_space','feature_id','tf_stable_id','checkpoint_sha256')},
                    'commit':record(ROOT,directory/'COMMIT.json'),'result':artifact('result.json'),
                    'edge_axes':artifact('edge_axes.json'),'vx_edges':artifact('vx_edges.npz'),'vy_edges':artifact('vy_edges.npz'),
                    'parent_baseline_dependency':value['parent_baseline_dependency'],
                    'qualification':{c:qualify(qrows,task,c) for c in ('vx','vy')}}
                key=(stage,task['input_space'],task['feature_id'])
                entry=inputs.setdefault(key,{'stage':stage,'input_space':task['input_space'],'feature_id':task['feature_id'],
                    'tf_stable_id':task['tf_stable_id'],'seed_tasks':[]})
                entry['seed_tasks'].append(item);census.append({'task_id':task['task_id'],'stage':stage,'commit':item['commit']})
    if len(census)!=4752 or any(sorted(r['task']['seed'] for r in x['seed_tasks'])!=[0,1,2] for x in inputs.values()):raise ValueError('Complete all3seed input families required')
    new(output/'complete_commit_census.json',{'precision_plan_sha256':plan['precision_plan_sha256'],'complete_tasks':4752,'census':census,'scientific_pass':False})
    summaries=[];consensus=[];group_axes={};tasks_by_id={t['task_id']:t for s in STAGES for t in books[s]['tasks']}
    for key,entry in inputs.items():
        channel_means={c:[] for c in ('vx','vy')};axes_by_seed={}
        for item in sorted(entry['seed_tasks'],key=lambda x:x['task']['seed']):
            task=tasks_by_id[item['task']['task_id']];axes=read(checked(ROOT,item['edge_axes']))
            validate_axes(axes,expected_axes(original,frame,task));axes_by_seed[task['seed']]=axes
            axis_key=task['stage']+'__seed'+str(task['seed'])
            if axis_key in group_axes and canonical(group_axes[axis_key])!=canonical(axes['groups']):raise ValueError('Groups differ across inputs within same sample/seed')
            group_axes[axis_key]=axes['groups']
            for channel in ('vx','vy'):
                source=checked(ROOT,item[channel+'_edges']);arrays=parse_edge_payload(source,axes,task)
                channel_means[channel].append(arrays['weighted_signed_mean'])
                for group_index,group in enumerate(axes['groups']):
                    values=arrays['weighted_signed_mean'][group_index];finite=np.isfinite(values)
                    score=float(np.sqrt(np.sum(values[finite]**2,dtype=np.float64))) if finite.any() else None
                    summaries.append({'stage':task['stage'],'input_space':task['input_space'],'feature_id':task['feature_id'],'tf_stable_id':task['tf_stable_id'],
                        'seed':task['seed'],'sample_id':task['sample_id'],'channel':channel,'group_kind':group['kind'],'group_name':group['name'],
                        'sample_n':group['sample_n'],'effect_frame_domain_n':group['effect_frame_domain_n'],'sample_weight_sum':group['sample_weight_sum'],
                        'target_count':len(values),'positive_targets':int((values>0).sum()),'negative_targets':int((values<0).sum()),
                        'zero_targets':int((values==0).sum()),'NA_targets':int((~finite).sum()),'weighted_mean_vector_L2':score,
                        'descriptive_rank':None,'descriptive_percentile':None,'rank_valid_regulator_count':None,'rank_full_family_regulator_count':None,
                        'qualification_status':item['qualification'][channel]['status'],'patient_P_value':None,'scientific_pass':False})
                checked(ROOT,item[channel+'_edges']) # ensure immutable source persisted through parse
        reference_groups=axes_by_seed[0]['groups']
        for channel,matrices in channel_means.items():
            tensor=np.stack(matrices);assert tensor.shape[0]==3
            # Structural group identity retained across seeds, while n/masses
            # may differ for original seed-specific time-window strata.
            for seed,axes in axes_by_seed.items():
                if [(g['kind'],g['name']) for g in axes['groups']]!=[(g['kind'],g['name']) for g in reference_groups]:raise ValueError('Group names differ across seeds')
            for i,group in enumerate(reference_groups):
                values=tensor[:,i,:];nfinite=np.isfinite(values).sum(axis=0);pos=(values>0).sum(axis=0);neg=(values<0).sum(axis=0);zero=(values==0).sum(axis=0)
                consensus.append({'stage':entry['stage'],'input_space':entry['input_space'],'feature_id':entry['feature_id'],'channel':channel,
                    'group_kind':group['kind'],'group_name':group['name'],'target_count':values.shape[1],
                    'all3_positive_targets':int((pos==3).sum()),'all3_negative_targets':int((neg==3).sum()),'all3_zero_targets':int((zero==3).sum()),
                    'mixed_sign_or_zero_complete_targets':int(((nfinite==3)&(pos!=3)&(neg!=3)&(zero!=3)).sum()),
                    'targets_with_any_NA':int((nfinite<3).sum()),'seeds_are_independent_donors':False,'patient_P_value':None,'scientific_pass':False})
    ranks=defaultdict(list)
    for row in summaries:ranks[tuple(row[k] for k in ('stage','input_space','seed','channel','group_kind','group_name'))].append(row)
    for family in ranks.values():
        valid=[r for r in family if r['weighted_mean_vector_L2'] is not None]
        values=np.array([r['weighted_mean_vector_L2'] for r in valid]);ascending=rankdata(values,method='average')
        for row in family:row.update(rank_valid_regulator_count=len(valid),rank_full_family_regulator_count=len(family))
        for row,rank in zip(valid,ascending):row.update(descriptive_rank=float(len(valid)+1-rank),descriptive_percentile=float((rank-1)/(len(valid)-1)) if len(valid)>1 else None)
    for name,rows in [('all_input_strength_ranks.tsv',summaries),('all_input_direction_consistency.tsv',consensus)]:
        with (output/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    index={'schema':'dynavelo_complete_precision_grn_backed_index_v1','created_utc':now(),'status':'ALL4668_PLUS84_ORIGINAL_COMMITS_AND_PARENTS_VERIFIED',
        'precision_plan':record(ROOT,path),'precision_plan_sha256':plan['precision_plan_sha256'],'source':record(ROOT,Path(__file__)),
        'original_scope_plan':plan['original_scope_plan'],'scope_frame':plan['scope_frame'],'shortlist':plan['shortlist'],
        'full_family_counts':{'grn':4668,'full-cell-grn':84},'parent_baselines_complete':12,'coverage_records':coverage_records,
        'actual_qualification_review':review_record,'epsilon_qualification_coverage':record(ROOT,output/'epsilon_qualification_coverage.json'),
        'qualification_map':qrecord,'qualification_map_missing_policy':'NOT_TESTED; never infer from external P or remove from full family',
        'rna_target_axis':[{'feature_id':r['feature_id'],'stable_id':r['stable_id']} for r in original['context_plan']['spec']['genes']],
        'motif_target_axis':original['context_plan']['spec']['motifs'],'group_axes':group_axes,'inputs':list(inputs.values()),
        'original_negative_zero_NA_stats_retained_in_backing_artifacts':True,'all_original11_statistics_queryable':True,
        'frozen20_candidates':shortlist['selected'],'candidate_replacement':False,'new_external_validation_selected':False,
        'rank_definition':'Perseed,stage,inputspace,channel,referencegroup L2 norm of all finite weighted signed target means; average ties; NA stays in complete family',
        'reference_time_windows':'Original FP32 frozen global windows; not recomputed from FP64 perturbations',
        'patient_P_values_computed':False,'statistical_independent_units_added':0,'scientific_pass':False,'seconds':time.monotonic()-started}
    new(output/'complete_backed_grn_index.json',index)
    return {'status':index['status'],'input_families':len(inputs),'complete_tasks':4752,'strength_rows':len(summaries),'consensus_rows':len(consensus),'seconds':index['seconds'],'scientific_pass':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output');parser.add_argument('--qualification-map');parser.add_argument('--qualification-map-sha')
    args=parser.parse_args();p=Path(args.plan)
    if args.execute:
        if not args.output:parser.error('--execute requires a new --output')
        result=aggregate(p,args.plan_sha,Path(args.output),qualification_map=Path(args.qualification_map) if args.qualification_map else None,qualification_sha=args.qualification_map_sha)
    else:result=metadata_dry_run(p,args.plan_sha)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
