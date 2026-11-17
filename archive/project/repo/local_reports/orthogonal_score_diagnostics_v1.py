"""Local post-hoc score diagnostics; read existing files only, never fit models.

A: retain both the original raw top50 and original qualified subsets, separately.
B: use the existing ALL1191 PRIMARY192 matrices; never call them full1884.
Requires only NumPy and SciPy. All output is confined to a NEW report directory.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.stats import beta, hypergeom, rankdata

REPORTS = 'results/reports/'
READOUT = REPORTS + 'orthogonal_readout_results_20261116_v1/'
PREP = REPORTS + 'orthogonal_readout_resume_preparation_20261116_v1/'
BUNDLE = REPORTS + 'primary_global_diagnostic_cloud_20261116_v1/input_package/'
DEFAULT_OUT = REPORTS + 'orthogonal_score_diagnostics_20261116_v1'
BH_FORMULA = 'q_(i)=min(1,min_{j>=i}(m*p_(j)/j)); sorted ascending; m=20 per fixed edge-set family'
HYPERGEOM_FORMULA = 'X_s~Hypergeom(N_s,K_CPU_s,n_GRN_s); P=Pr(sum_s X_s>=observed); 25 fixed strata, PMF convolution'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class Inputs:
    def __init__(self, root):
        self.root, self.records = root, {}

    def path(self, relative, expected=None):
        p = (self.root / relative).resolve()
        require(p.is_relative_to(self.root) and p.is_file(), 'Missing/outside input: ' + str(p))
        rec = {'relative_path': p.relative_to(self.root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
        if expected:
            require(rec['bytes'] == expected['bytes'] and rec['sha256'] == expected['sha256'], 'Frozen input hash mismatch: ' + str(p))
        if relative in self.records:
            require(self.records[relative] == rec, 'Input changed during read')
        self.records[relative] = rec
        return p

    def json(self, relative, expected=None):
        return json.loads(self.path(relative, expected).read_text(encoding='utf-8-sig'))

    def tsv(self, relative, expected=None):
        with self.path(relative, expected).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f, delimiter='\t'))

    def verify_unchanged(self):
        for rec in self.records.values():
            p = self.root / rec['relative_path']
            require(p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256'], 'Upstream changed during diagnostic: ' + str(p))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_table(path, rows, fields=None):
    with path.open('x', encoding='utf-8-sig' if path.suffix == '.csv' else 'utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), delimiter=',' if path.suffix == '.csv' else '\t')
        writer.writeheader()
        writer.writerows(rows)


def adjust_bh(pvalues):
    p = np.asarray(pvalues, float)
    require(p.ndim == 1 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid P family')
    order = np.argsort(p, kind='stable')
    q = np.empty_like(p)
    q[order] = np.minimum(1., np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    return q


def top_sets(score, self_index, k=50):
    """Exact k by score then target index, plus inclusive-boundary sensitivity."""
    valid = np.flatnonzero(np.isfinite(score) & (score > 0))
    valid = valid[valid != self_index]
    order = valid[np.argsort(-score[valid], kind='stable')]
    exact = order[:k]
    tied = order if not len(exact) else order[score[order] >= score[exact[-1]]]
    return set(map(int, exact)), set(map(int, tied))


def conditional_overlap(a, b, background, strata):
    a, b, bg = set(a), set(b), set(background)
    require(a <= bg and b <= bg, 'Selected edge outside background')
    observed, expected, pmf, margins = len(a & b), 0., np.array([1.]), []
    for s in range(25):
        layer = {i for i in bg if strata[i] == s}
        n, ka, kb = len(layer), len(a & layer), len(b & layer)
        margins.append({'stratum': s, 'background_n': n, 'cpu_n': ka, 'grn_n': kb})
        if n:
            expected += ka * kb / n
        if n and ka and kb:
            local = hypergeom.pmf(np.arange(min(ka, kb) + 1), n, ka, kb)
            require(abs(local.sum() - 1.) < 1e-10, 'Hypergeometric PMF mass failure')
            pmf = np.convolve(pmf, local)
    require(abs(pmf.sum() - 1.) < 1e-9, 'Convolution mass failure')
    return {'overlap_n': observed, 'expected_overlap': expected,
            'observed_over_expected': observed / expected if expected else None,
            'p_raw_25strata': min(1., float(pmf[observed:].sum())),
            'jaccard': observed / len(a | b) if a | b else None,
            'cpu_set_n': len(a), 'grn_set_n': len(b), 'background_n': len(bg)}, pmf, margins


def column_transform(absolute_matrix):
    """Aggregate first, then normalize across all1191 inputs within each target."""
    d = np.asarray(absolute_matrix, float)
    require(d.ndim == 2 and np.isfinite(d).all() and (d >= 0).all(), 'Finite nonnegative derivative matrix required')
    sums = d.sum(axis=0)
    l1 = np.divide(d, sums[None, :], out=np.zeros_like(d), where=sums[None, :] > 0)
    pct = np.zeros_like(d)
    for start in range(0, d.shape[1], 128):
        block = d[:, start:start + 128]
        pct[:, start:start + 128] = (rankdata(block, axis=0, method='average') - .5) / len(d)
    pct[:, sums == 0] = 0.
    return l1, pct, sums


def spearman_only(a, b):
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'No silent deletion for correlation')
    x, y = rankdata(a, method='average'), rankdata(b, method='average')
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt(np.sum(x*x) * np.sum(y*y))
    return float(np.sum(x*y) / den) if den else None


def weight_matrix(cpu_sets, grn_sets, n_targets):
    """W[t,u]=number of candidates with CPU target t and GRN target u."""
    w = np.zeros((n_targets, n_targets), dtype=np.int16)
    for a, b in zip(cpu_sets, grn_sets):
        w[np.ix_(sorted(a), sorted(b))] += 1
    return w


def shared_permutation_pool(configs, strata, fixed_targets, repeats, seed):
    """Shared target-label permutations preserve cross-candidate dependencies.

    Pin all20 candidate target columns, so every candidate's forbidden self edge
    remains forbidden. Shuffle all other columns within frozen expression strata.
    Each permutation is shared across candidates AND score/edge-set configurations.
    """
    n = len(strata)
    fixed = set(fixed_targets)
    groups = [np.array([i for i in range(n) if strata[i] == s and i not in fixed], dtype=int) for s in range(25)]
    weights = {key: weight_matrix(a, b, n) for key, (a, b) in configs.items()}
    active = {key: np.flatnonzero(w.sum(axis=1)) for key, w in weights.items()}
    observed = {key: int(np.trace(w)) for key, w in weights.items()}
    totals = {key: np.empty(repeats, dtype=np.int32) for key in configs}
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, 256):
        size = min(256, repeats-start)
        perm = np.tile(np.arange(n, dtype=np.int32), (size, 1))
        for group in groups:
            if len(group) > 1:
                perm[:, group] = rng.permuted(np.broadcast_to(group, (size, len(group))), axis=1)
        for key, w in weights.items():
            idx = active[key]
            totals[key][start:start+size] = w[idx[None, :], perm[:, idx]].sum(axis=1)
    results, histograms = [], []
    for key, w in weights.items():
        fixed_overlap = sum(int(w[i, i]) for i in fixed)
        expected = fixed_overlap + sum(float(w[np.ix_(g, g)].sum()) / len(g) for g in groups if len(g))
        values = totals[key]
        hits = int((values >= observed[key]).sum())
        ci = [0. if hits == 0 else float(beta.ppf(.025, hits, repeats-hits+1)),
              1. if hits == repeats else float(beta.ppf(.975, hits+1, repeats-hits))]
        results.append({'configuration': key, 'pooled_overlap_n': observed[key],
                        'pooled_expected_shared_permutation_null': expected,
                        'pooled_observed_over_expected': observed[key]/expected if expected else None,
                        'pooled_p_shared_stratified_permutation': (hits+1)/(repeats+1),
                        'exceedances_ge_observed': hits, 'permutations': repeats, 'random_seed': seed,
                        'mc_tail_probability_clopper_pearson95_low': ci[0], 'mc_tail_probability_clopper_pearson95_high': ci[1],
                        'permutation_mean': float(values.mean()), 'permutation_std_ddof0': float(values.std()),
                        'fixed_candidate_target_columns_n': len(fixed), 'fixed_overlap_n': fixed_overlap,
                        'permuted_target_columns_n': n-len(fixed),
                        'null': 'shared column permutation within 25 strata; all20 self-target columns fixed; preserves row margins and cross-candidate membership patterns',
                        'status': 'POST_HOC_ALGORITHMIC_DIAGNOSTIC_ONLY'})
        counts = np.bincount(values)
        histograms.extend({'configuration': key, 'pooled_overlap_n': i, 'permutation_count': int(v)} for i, v in enumerate(counts) if v)
    return results, histograms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument('--permutations', type=int, default=99999)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    start = time.monotonic()
    root = args.project_root.resolve()
    out = (root / args.output_dir).resolve()
    require(out.is_relative_to(root / 'results/reports') and not out.exists(), 'Use a NEW report directory; no overwrite')
    require(args.permutations >= 999 and args.seed >= 0, 'At least999 permutations and nonnegative seed')
    inputs = Inputs(root)
    complete = inputs.json(READOUT + 'COMPLETE.json')
    prep = inputs.json(PREP + 'prepared_inputs.json', complete['prepared'])
    records = {r['relative_path']: r for r in complete['outputs'] + prep['inputs']}
    primary = inputs.json(READOUT + 'primary_results_all20.json', records[READOUT + 'primary_results_all20.json'])
    auxiliary = inputs.json(READOUT + 'auxiliary_official_importance_all20.json', records[READOUT + 'auxiliary_official_importance_all20.json'])
    shortlist_path = REPORTS + 'a100_production_20261114/post_screen_v1/locked_scope_shortlist.json'
    shortlist = inputs.json(shortlist_path, records[shortlist_path])['selected']
    genes = [x['feature_id'] for x in shortlist]
    by_gene = {x['gene']: x for x in prep['candidates']}
    primary_by_gene = {x['gene']: x for x in primary}
    aux_by_gene = {x['gene']: x for x in auxiliary}
    require(len(genes) == len(set(genes)) == 20 and set(genes) == set(by_gene) == set(primary_by_gene), 'All fixed20 required')
    candidates = [by_gene[g] for g in genes]
    moments = inputs.tsv(prep['target_moments']['relative_path'], prep['target_moments'])
    require([int(x['target_index']) for x in moments] == list(range(3006)), 'Target moments axis differs')
    strata = np.array([int(x['stratum']) for x in moments])
    target_ids = [x['stable_id'] for x in moments]
    target_names = [x['symbol'] for x in moments]
    self_indices = [x['target_index'] for x in candidates]
    cpu_sets = {
        'raw1000': [set(x['cpu_raw_top50_with_ties']) - {x['target_index']} for x in candidates],
        'qualified952': [set(x['cpu_supported_nonself_target_indices']) for x in candidates],
    }
    raw_edges_path = REPORTS + 'cpu_locked20_primary_axis_review_20261116_v1/all20_raw_primary_top50_with_flags.tsv'
    raw_edges = inputs.tsv(raw_edges_path, records[raw_edges_path])
    require(len(raw_edges) == 1000, 'Unexpected raw edge count')
    for candidate, a, qualified in zip(candidates, cpu_sets['raw1000'], cpu_sets['qualified952']):
        own = [r for r in raw_edges if r['input_gene'] == candidate['gene']]
        require(a == {int(r['target_index']) for r in own}, 'Raw edges would change')
        require(qualified == {int(r['target_index']) for r in own if r['numerically_supported_directional_nonself'] == 'True'}, 'Qualified edges would change')
    scores_path = READOUT + 'all20_all3006_target_scores.tsv'
    scores = inputs.tsv(scores_path, records[scores_path])
    score_arrays = {mode: np.full((20,3006), np.nan) for mode in ('official_raw_median', 'normalized_median')}
    score_lookup = {}
    gene_index = {g:i for i,g in enumerate(genes)}
    for row in scores:
        i, j = gene_index[row['candidate_gene']], int(row['target_index'])
        require((i,j) not in score_lookup and row['target_stable_id'] == target_ids[j], 'Score identity/axis mismatch')
        require(row['candidate_stable_id'] == candidates[i]['stable_id'], 'Candidate stable ID differs')
        score_lookup[i,j] = row
        if j != self_indices[i]:
            require(row['status'] == 'COMPUTABLE', 'Unresolved nonself GRN score')
            seed_raw = np.array([float(row[f'importance_seed{s}']) for s in range(3)])
            seed_trees = np.array([float(row[f'n_estimators_seed{s}']) for s in range(3)])
            require((seed_trees > 0).all(), 'Invalid tree count')
            for mode, expected in [('official_raw_median', np.median(seed_raw)), ('normalized_median', np.median(seed_raw/seed_trees))]:
                value = float(row[mode])
                require(np.isfinite(value) and value >= 0 and np.isclose(value, expected, rtol=1e-13, atol=1e-15), 'Stored score formula differs')
                score_arrays[mode][i,j] = value
    require(len(score_lookup) == 20*3006, 'Incomplete20x3006 score table')
    grn_sets = {mode: [top_sets(score, own)[1] for score, own in zip(array, self_indices)] for mode,array in score_arrays.items()}
    a_rows, strata_rows, independent_pooled, configs = [], [], {}, {}
    for edge_label, a_sets in cpu_sets.items():
        for mode, b_sets in grn_sets.items():
            config = edge_label + '__' + mode
            configs[config] = (a_sets,b_sets)
            group, total_pmf, observed_sum, expected_sum = [], np.array([1.]), 0, 0.
            for i, (candidate,a,b) in enumerate(zip(candidates,a_sets,b_sets)):
                stats, pmf, margins = conditional_overlap(a,b,set(range(3006))-{self_indices[i]},strata)
                if edge_label == 'qualified952':
                    old = (aux_by_gene if mode == 'official_raw_median' else primary_by_gene)[candidate['gene']]
                    require(a == set(old['cpu_target_indices']) and b == set(old['algorithm_target_indices']), 'Frozen membership not reproduced')
                    require(stats['overlap_n'] == old['observed_overlap'] and np.isclose(stats['p_raw_25strata'],old['p_raw'],rtol=1e-11,atol=1e-14), 'Frozen P not reproduced')
                group.append({'candidate_order':i+1,'candidate_gene':candidate['gene'],'candidate_stable_id':candidate['stable_id'],
                              'configuration':config,'cpu_edge_set':edge_label,'grn_score':mode,
                              'derivative_scope':'CPU_FULL1884_EFFECT_POPULATION_ALL_VX',**stats,
                              'shared_targets':';'.join(target_names[j] for j in sorted(a&b)),
                              'status':'POST_HOC_ALGORITHMIC_DIAGNOSTIC_ONLY'})
                strata_rows.extend({'configuration':config,'candidate_gene':candidate['gene'],**m} for m in margins)
                total_pmf=np.convolve(total_pmf,pmf)
                observed_sum+=stats['overlap_n'];expected_sum+=stats['expected_overlap']
            for row,q in zip(group,adjust_bh([r['p_raw_25strata'] for r in group])):
                row['q_BH_within20'] = float(q)
                row['q_BY_within20_sensitivity'] = min(1.,float(q)*sum(1/k for k in range(1,21)))
                row['BH_formula'] = BH_FORMULA
            a_rows.extend(group)
            independent_pooled[config]={'pooled_p_independent_candidate_convolution_SENSITIVITY':min(1.,float(total_pmf[observed_sum:].sum())),
                                       'pooled_expected_independent_margins':expected_sum,
                                       'independence_caveat':'Candidate sets share targets; this P assumes independent rows and is NOT the preferred pooled test'}
    require(sum(map(len,cpu_sets['raw1000'])) == 1000 and sum(map(len,cpu_sets['qualified952'])) == 952,'Frozen edge totals differ')
    print('A: verified frozen sets and original P values; calculating shared stratified pooled null.',flush=True)
    pooled, histograms=shared_permutation_pool(configs,strata,self_indices,args.permutations,args.seed)
    for row in pooled:
        row.update(independent_pooled[row['configuration']])

    contract=inputs.json(BUNDLE+'contract.json')
    require(contract['scope']=='A100_PRIMARY192_ALL1191_RNA_INPUTS_EFFECT_POPULATION_ALL_VX','Do not silently change B scope')
    axes=inputs.json(BUNDLE+contract['axes']['path'],contract['axes'])
    provenance=inputs.json(BUNDLE+contract['provenance']['path'],contract['provenance'])
    require([x['stable_id'] for x in axes['targets']]==target_ids,'A100/GRN target axes disagree')
    grn_axis_path=REPORTS+'orthogonal_GRN_preparation_20261116_v1/native_pilot_v1/axes.json'
    grn_axes=inputs.json(grn_axis_path,records[grn_axis_path])
    require([x['stable_id'] for x in axes['inputs']]==grn_axes['predictor_stable_ids'],'All1191 predictor identities/order differ')
    index_by_id={x['stable_id']:x['index'] for x in axes['inputs']}
    selected_indices=[index_by_id[x['stable_id']] for x in candidates]
    require(len(axes['inputs'])==1191 and len(axes['targets'])==3006,'All1191x3006 required')
    matrices=[]
    require({r['seed'] for r in contract['matrices']}=={0,1,2},'Three technical seeds required')
    for rec in sorted(contract['matrices'],key=lambda x:x['seed']):
        array=np.load(inputs.path(BUNDLE+rec['path'],rec),mmap_mode='r',allow_pickle=False)
        require(array.shape==(1191,3006) and array.dtype==np.float64 and np.isfinite(array).all(),'Invalid existing derivative matrix')
        matrices.append(array)
    stack=np.stack(matrices)
    signed=np.median(stack[:,selected_indices,:],axis=0)
    np.abs(stack,out=stack)
    magnitude=np.median(stack,axis=0,overwrite_input=True)
    del stack,matrices
    l1,percentile,column_sums=column_transform(magnitude)
    variants={'B0_raw_abs_PRIMARY192':magnitude[selected_indices],
              'B1_target_column_L1_PRIMARY192':l1[selected_indices],
              'B2_target_column_percentile_PRIMARY192':percentile[selected_indices]}
    del l1,percentile
    print('B: existing PRIMARY192 matrices normalized within target across all1191; no derivatives recomputed.',flush=True)
    b_rows, b_top, b_members={},[],{}
    frequency={label:np.zeros(3006,dtype=int) for label in variants}
    for variant,array in variants.items():
        for i,candidate in enumerate(candidates):
            exact,tied=top_sets(array[i],self_indices[i])
            b_members[variant,i]=(exact,tied)
            frequency[variant][list(exact)]+=1
            background=np.arange(3006)!=self_indices[i]
            require(int(background.sum())==3005,'Spearman must include all3005 nonself targets')
            for mode,grn_array in score_arrays.items():
                b=grn_sets[mode][i]
                row={'candidate_order':i+1,'candidate_gene':candidate['gene'],'candidate_stable_id':candidate['stable_id'],
                     'derivative_variant':variant,'derivative_scope':'A100_PRIMARY192_NOT_CPU_FULL1884',
                     'grn_score':mode,'derivative_top_exact_n':len(exact),'derivative_top_boundary_inclusive_n':len(tied),
                     'boundary_ties_excluded_from_exact50_n':len(tied-exact),'grn_top50_ties_n':len(b),
                     'overlap_exact50_n':len(exact&b),'overlap_boundary_inclusive_n':len(tied&b),
                     'jaccard_exact50':len(exact&b)/len(exact|b) if exact|b else None,
                     'spearman_all3005':spearman_only(array[i,background],grn_array[i,background]),'spearman_target_n':3005,
                     'score_unique_nonself_n':int(len(np.unique(array[i,background]))),
                     'grn_nonself_zero_n':int((grn_array[i,background]==0).sum()),
                     'shared_targets_exact50':';'.join(target_names[j] for j in sorted(exact&b)),
                     'qualification_policy':contract['qualification_policy'],
                     'status':'POST_HOC_DESCRIPTIVE_NO_NEW_B_P_VALUES'}
                b_rows[variant,i,mode]=row
            ordering=sorted(tied,key=lambda j:(-array[i,j],j))
            for rank,j in enumerate(ordering,1):
                b_top.append({'candidate_gene':candidate['gene'],'target_gene':target_names[j],'target_stable_id':target_ids[j],
                              'target_index':j,'variant':variant,'deterministic_rank':rank,'in_exact50':j in exact,
                              'in_boundary_inclusive':True,'derivative_score':float(array[i,j]),
                              'original_abs_median3':float(magnitude[selected_indices[i],j]),'signed_median3':float(signed[i,j]),
                              'grn_official_raw_median':float(score_arrays['official_raw_median'][i,j]),
                              'grn_normalized_median':float(score_arrays['normalized_median'][i,j]),
                              'in_grn_raw_top50':j in grn_sets['official_raw_median'][i],
                              'in_grn_normalized_top50':j in grn_sets['normalized_median'][i],
                              'qualification':'PRIMARY192_ALL_NOT_TESTED_original_HOLD_unchanged'})
    compare=[]
    a_lookup={(r['configuration'],r['candidate_gene']):r for r in a_rows}
    for i,gene in enumerate(genes):
        a=a_lookup['raw1000__official_raw_median',gene]
        gated=a_lookup['qualified952__official_raw_median',gene]
        row={'candidate_gene':gene,
             'A_raw_edges_old_tree_normalized_overlap':a_lookup['raw1000__normalized_median',gene]['overlap_n'],
             'A_raw_edges_official_overlap':a['overlap_n'],'A_raw_edges_p25':a['p_raw_25strata'],'A_raw_edges_BH_q':a['q_BH_within20'],
             'A_qualified_edges_official_overlap':gated['overlap_n'],'A_qualified_edges_p25':gated['p_raw_25strata'],'A_qualified_edges_BH_q':gated['q_BH_within20']}
        for short,variant in zip(('B0_raw192','B1_colL1_192','B2_colPercentile_192'),variants):
            b=b_rows[variant,i,'official_raw_median']
            row[short+'_overlap']=b['overlap_exact50_n'];row[short+'_rho3005']=b['spearman_all3005']
            row[short+'_boundary_set_n']=b['derivative_top_boundary_inclusive_n']
        compare.append(row)
    frequencies=[]
    original_freq=np.zeros(3006,dtype=int)
    qualified_freq=np.zeros(3006,dtype=int)
    for a in cpu_sets['raw1000']: original_freq[list(a)]+=1
    for a in cpu_sets['qualified952']: qualified_freq[list(a)]+=1
    for j in range(3006):
        frequencies.append({'target_index':j,'target_gene':target_names[j],'target_stable_id':target_ids[j],
                            'frozen_mean':float(moments[j]['mean']),'frozen_population_variance':float(moments[j]['population_variance_ddof0']),
                            'frozen_stratum':int(strata[j]),'A_raw1000_occurrence':int(original_freq[j]),
                            'A_qualified952_occurrence':int(qualified_freq[j]),
                            **{label+'_exact50_occurrence':int(freq[j]) for label,freq in frequency.items()},
                            'PRIMARY192_column_sum_abs_median3':float(column_sums[j])})
    summary={'status':'COMPLETE_POST_HOC_DIAGNOSTIC','scientific_pass':False,'derivative_recomputed':False,'grnboost2_refitted':False,
             'A_original_raw_edges_n':int(original_freq.sum()),'A_original_distinct_targets':int((original_freq>0).sum()),
             'A_targets_present_in_all20':[target_names[j] for j in np.flatnonzero(original_freq==20)],
             'A':{},'B':{},'B_scope_warning':'Only ALL1191 PRIMARY192 matrix exists in the verified input package; no full1884 all1191 substitute or CPU-A100 equivalence claimed.',
             'B_qualification_policy':contract['qualification_policy'],'CPU_A100_bridge':provenance['CPU_to_A100_bridge']}
    for config in configs:
        rows=[r for r in a_rows if r['configuration']==config]
        summary['A'][config]={'overlap_sum':sum(r['overlap_n'] for r in rows),'expected_sum':sum(r['expected_overlap'] for r in rows),
                              'BH_q_lt_005_n':sum(r['q_BH_within20']<.05 for r in rows),'min_BH_q':min(r['q_BH_within20'] for r in rows),
                              'min_p25':min(r['p_raw_25strata'] for r in rows),
                              'pooled':[r for r in pooled if r['configuration']==config][0]}
    for variant in variants:
        rows=[b_rows[variant,i,'official_raw_median'] for i in range(20)]
        summary['B'][variant]={'overlap_sum_vs_official_raw':sum(r['overlap_exact50_n'] for r in rows),
                               'boundary_inclusive_overlap_sum':sum(r['overlap_boundary_inclusive_n'] for r in rows),
                               'distinct_top50_targets':int((frequency[variant]>0).sum()),
                               'targets_present_all20_n':int((frequency[variant]==20).sum()),
                               'median_spearman3005':float(np.median([r['spearman_all3005'] for r in rows])),
                               'min_spearman3005':min(r['spearman_all3005'] for r in rows),
                               'max_spearman3005':max(r['spearman_all3005'] for r in rows)}
    methods={
        'analysis_status':'Post hoc sensitivity diagnostics requested after original null results; do not replace frozen confirmatory test or claim newly independent validation.',
        'A_fixed_edge_sets':{'raw1000':'All original20x50 edges; retain unsupported flags; no new filtering/refill. Each row excludes self; actual original sets have no self.',
                             'qualified952':'Exactly original supported directional nonself edge subsets,44-50 per candidate; no refill.'},
        'GRN_baseline':'Existing GRNBoost2 arboreto0.1.6, sklearn1.2.2, seeds0/1/2,6035-cell RNA,1191 eligible predictors,3006 targets; no fit called.',
        'GRN_official_score':'median_s(official_importance_s); NOT median(importance_s/n_estimators_s)',
        'A_primary_P':HYPERGEOM_FORMULA,'A_strata':'Existing frozen mean x population variance quintiles;25 labels including empty strata; each row3005 nonself targets.',
        'BH':BH_FORMULA,'BH_limit':'20-test family for each displayed configuration, not a correction for trying score variants. BH assumes independence or suitable positive dependence; BY sensitivity also shown.',
        'BY_formula':'min(1,q_BH*sum_{k=1}^20(1/k)); dependence-robust sensitivity',
        'pooled_primary':'T=sum_i|CPU_i intersect GRN_i|; shared target permutations within25 strata, pin20 candidate target columns to preserve self exclusions; p=(1+number(T_perm>=T_observed))/(B+1). This conditions on the fixed columns and has its own reported expectation.',
        'pooled_sensitivity':'Convolve all20 candidate PMFs; requires independence across candidate rows, which overlapping target sets do not establish; supplementary only.',
        'B_scope':'Existing A100 PRIMARY192, all1191x3006 weighted_signed_mean, three technical seeds. B0 and B1/B2 use identical cells, axes and seed aggregation. No original full1884 endpoint substitution.',
        'B_base_magnitude':'D_ij=median_{s=0,1,2}|existing weighted_signed_mean_s[i,j]|; aggregate before normalization; no signed cancellation.',
        'B1_column_L1':'L_ij=D_ij/sum_{k=1}^{1191}D_kj; zero-sum columns ->0. All1191 including diagonal enter denominator; self edges excluded only when selecting/comparing targets.',
        'B2_column_percentile':'P_ij=(average_rank_across1191(D_ij)-0.5)/1191; ties use average rank; all-zero columns ->0.',
        'B_top50':'Exact50 positive scores per candidate, exclude self, descending score then ascending original target index. Also report ALL exact boundary ties and inclusive-overlap sensitivity; ties never broken using GRN results.',
        'B_Spearman':'Pearson correlation of average ranks of the derivative magnitude score and GRN importance across ALL3005 nonself targets, including GRN zeros; no score-based deletion; constant rank vector->NA; no correlation P computed.',
        'direction_limit':'Magnitude/importance ranks are unsigned; rho sign is rank concordance, not activation/repression. Column specificity is a constructed score, not proven biological specificity.',
        'B_qualifications':contract['qualification_policy'],
        'sources':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html',
                   'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html',
                   'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html']}
    inputs.verify_unchanged()
    out.mkdir(parents=True,exist_ok=False)
    write_table(out/'comparison_all20.csv',compare)
    write_table(out/'A_all20_fixed_edges_comparison.csv',a_rows)
    write_table(out/'A_all25_stratum_margins.tsv',strata_rows)
    write_table(out/'A_pooled_enrichment.csv',pooled)
    write_table(out/'A_pooled_permutation_histograms.tsv',histograms)
    write_table(out/'B_all20_top50_and_spearman.csv',list(b_rows.values()))
    write_table(out/'B_selected_edges_with_boundary_ties.tsv',b_top)
    write_table(out/'target_occurrence_and_scale_diagnostics.csv',frequencies)
    # Long table includes all3005 targets per candidate, even zero-importance targets.
    def full_score_rows():
        for i,gene in enumerate(genes):
            for j in range(3006):
                if j==self_indices[i]:continue
                yield {'candidate_gene':gene,'target_index':j,'target_gene':target_names[j],'target_stable_id':target_ids[j],
                       **{label:float(array[i,j]) for label,array in variants.items()},
                       **{mode:float(array[i,j]) for mode,array in score_arrays.items()}}
    fields=['candidate_gene','target_index','target_gene','target_stable_id']+list(variants)+list(score_arrays)
    write_table(out/'B_all20_all3005_scores.tsv',full_score_rows(),fields)
    write_json(out/'methods_and_formulas.json',methods)
    write_json(out/'summary.json',summary)
    lines=['# 评分与靶基因尺度：事后诊断','',
           '只读取已有结果；没有重新计算导数或拟合GRNBoost2。原冻结检验结论不被覆盖。','',
           'A同时保留原始1000条边与原正式检验的952条合格边；每种口径分别固定边集比较两个GRN评分。',
           'B使用现有ALL1191的PRIMARY192矩阵。B0/B1/B2在同一矩阵上比较，不能冒充CPU全1884细胞归一化结果。',
           'B所有原HOLD/NOT_TESTED保留；Spearman与importance只表示无符号评分排序的一致性。','',
           '## A：25层超几何与BH（每组20个候选）','',
           '| 固定边集与GRN评分 | 总交集 | BH<0.05候选数 | 最小BH q | 保留依赖的pooled P |','|---|---:|---:|---:|---:|']
    for config,x in summary['A'].items():
        lines.append(f"| {config} | {x['overlap_sum']} | {x['BH_q_lt_005_n']} | {x['min_BH_q']:.6g} | {x['pooled']['pooled_p_shared_stratified_permutation']:.6g} |")
    lines+=['','总体P使用固定种子的99999次共同靶标分层置换（实际次数见表），保留候选间共享靶标结构，固定20个候选基因靶列以保留非自边约束。',
            '独立候选PMF卷积的总体P另列敏感性结果；不得忽略其独立性假设。BH为每个20候选组的事后校正，未校正所有分析策略选择。','',
            '## B：同一PRIMARY192矩阵内比较（GRN官方原始importance）','',
            '| 导数评分 | Top50交集总数 | Top50覆盖靶基因数 | Spearman中位数 |','|---|---:|---:|---:|']
    for label,x in summary['B'].items():
        lines.append(f"| {label} | {x['overlap_sum_vs_official_raw']} | {x['distinct_top50_targets']} | {x['median_spearman3005']:.6g} |")
    lines+=['','B1公式：D_ij / sum_k D_kj，其中D是3个seed绝对导数的中位数，k遍历1191个调控因子。',
            'B2公式：(列内平均秩-0.5)/1191。Top50并列按原target_index确定，另保留全部边界并列的敏感性比较。',
            'Spearman使用每个候选全3005个非自靶基因，包含GRN零分；没有只对Top50或非零importance计算。','',
            '完整20行总表：comparison_all20.csv。A详细表含80行（2边集×2评分×20）；B详细表120行（3变换×2评分×20）。',
            '全部公式、假设、文档来源见methods_and_formulas.json。输入前后SHA-256校验见audit.json。','',
            '重跑只运行本诊断脚本，并指定不存在的新输出目录；不会启动上游：','',
            '```powershell',f'& "{root / "repo/.venv_orthogonal_grn_v1/Scripts/python.exe"}" -B "{Path(__file__).resolve()}" --output-dir "results/reports/orthogonal_score_diagnostics_NEW"','```']
    with (out/'README_zh.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    inputs.verify_unchanged()
    audit={'status':'COMPLETE_UPSTREAM_UNCHANGED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'elapsed_seconds':time.monotonic()-start,'script_sha256':sha(__file__),'script_path':str(Path(__file__).resolve()),
           'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'cpu_threads':1,'processes':1,
           'derivatives_recomputed':False,'grnboost2_refitted':False,'inputs_before_after_sha256_identical':True,
           'inputs':list(inputs.records.values()),'permutations':args.permutations,'seed':args.seed,
           'rows':{'comparison':20,'A':len(a_rows),'B':len(b_rows),'B_full_target_scores':20*3005,'B_selected_edges_including_ties':len(b_top)},
           'outputs':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir())]}
    try:
        import psutil
        audit['process_peak_working_set_bytes']=psutil.Process().memory_info().peak_wset
    except (ImportError,AttributeError):pass
    write_json(out/'audit.json',audit)
    print(json.dumps({'status':audit['status'],'elapsed_seconds':audit['elapsed_seconds'],'output':str(out),'A':{k:{a:b for a,b in v.items() if a!='pooled'} for k,v in summary['A'].items()},'B':summary['B']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
