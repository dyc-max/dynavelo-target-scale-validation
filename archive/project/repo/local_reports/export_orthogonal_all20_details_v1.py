"""Export existing locked20 orthogonal results; standard library, no upstream writes.

P values are candidate-level set-concordance tests, never individual-edge tests.
The summary's representative edge is a display join chosen by ORIGINAL CPU rank.
All original CPU top50-with-ties rows are exported separately, without filtering.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import statistics
import sys

BASE = 'results/reports/'
PATHS = {
    'complete': BASE + 'orthogonal_readout_results_20261116_v1/COMPLETE.json',
    'primary_json': BASE + 'orthogonal_readout_results_20261116_v1/primary_results_all20.json',
    'primary_tsv': BASE + 'orthogonal_readout_results_20261116_v1/primary_results_all20.tsv',
    'target_scores': BASE + 'orthogonal_readout_results_20261116_v1/all20_all3006_target_scores.tsv',
    'prepared': BASE + 'orthogonal_readout_resume_preparation_20261116_v1/prepared_inputs.json',
    'shortlist': BASE + 'a100_production_20261114/post_screen_v1/locked_scope_shortlist.json',
    'cpu_edges': BASE + 'cpu_locked20_primary_axis_review_20261116_v1/all20_raw_primary_top50_with_flags.tsv',
    'plan': BASE + 'orthogonal_GRN_validation_plan_20261116_v1/prospective_method_plan_v1.md',
    'readout_source': 'repo/orthogonal_readout_resume_20261116_v1/orthogonal_readout_v1.py',
    'runtime_contract': BASE + 'orthogonal_GRN_preparation_20261116_v1/native_full_v1/contract_v2.json',
}
HOLM = 'p_Holm_(i) = min(1, max_{1<=j<=i} [(m-j+1)*p_(j)]); m=20; 按原始P升序后还原原顺序'
ALGORITHM_SCORE = 'median_seed[official_importance(seed,target)/n_estimators(seed,target)]'
RAW_TEST = '25层固定边际超几何PMF卷积的单侧上尾: P_raw=Pr(sum_s X_s >= 观察交集数)'
NULL_MODEL = '按冻结RNA目标表达均值×总体方差五分位分25层; X_s~Hypergeom(N_s,K_CPU_s,n_GRN_s); E=sum_s K_CPU_s*n_GRN_s/N_s'
REP_RULE = ('每个候选在既有原始CPU top50(含边界并列)表中选排名最高非自边供展示; '
            '同排名按target_index最小者展示; 所有原始边均在附表保留; 不按P或正交结果选边')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(data):
    return json.loads(data.decode('utf-8-sig'))


def read_tsv(data):
    return csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t')


def finite_float(text):
    number = float(text)
    require(math.isfinite(number), 'Nonfinite score in displayed upstream record')
    return number


def direction(values):
    if all(x > 0 for x in values):
        return '正'
    if all(x < 0 for x in values):
        return '负'
    if all(x == 0 for x in values):
        return '全零'
    return '混合符号或含零'


def edge_values(row):
    reference = [finite_float(row[f'raw_mean_seed{s}_route3']) for s in range(3)]
    all12 = [finite_float(row[f'raw_mean_seed{s}_route{r}']) for s in range(3) for r in range(4)]
    require(math.isclose(statistics.median(map(abs, reference)), finite_float(row['raw_reference_median_abs']),
                         rel_tol=1e-13, abs_tol=1e-18), 'Reference derivative summary differs')
    return reference, statistics.median(reference), direction(all12)


def holm_check(primary):
    # Recompute only to verify the reported adjustment; exported P values remain
    # the existing stored values, not a new statistical analysis.
    p = [1.0 if x['p_raw'] is None else float(x['p_raw']) for x in primary]
    require(len(p) == 20 and all(0 <= x <= 1 for x in p), 'Invalid twenty-test P family')
    running = 0.0
    for rank0, index in enumerate(sorted(range(20), key=lambda i: p[i])):
        running = min(1.0, max(running, (20 - rank0) * p[index]))
        row = primary[index]
        require(math.isclose(running, row['p_holm_adjustment_result'], rel_tol=1e-13, abs_tol=1e-15),
                'Stored Holm correction does not match the frozen formula')
        require(row['holm_missing_placeholder_used'] == (row['p_raw'] is None), 'Missing-P flag differs')


def write_table(path, rows, delimiter, excel_bom=False):
    require(bool(rows), 'No table rows')
    with path.open('x', encoding='utf-8-sig' if excel_bom else 'utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path,
                        default=Path(BASE + 'orthogonal_all20_details_export_20261116_v1'))
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out = out.resolve()
    require(out.is_relative_to(root) and not out.exists(), 'Use a fresh output directory inside the project')
    paths = {key: root / relative for key, relative in PATHS.items()}
    require(all(p.resolve().is_relative_to(root) and p.is_file() for p in paths.values()), 'Input path missing/outside project')
    require(all(not p.resolve().is_relative_to(out) for p in paths.values()), 'Output may not contain any upstream input')
    data = {key: p.read_bytes() for key, p in paths.items()}
    before = {key: digest(value) for key, value in data.items()}
    complete, prepared = json_bytes(data['complete']), json_bytes(data['prepared'])
    records = {x['relative_path']: x for x in complete['outputs'] + prepared['inputs']}
    for x in (complete['source'], complete['prepared']):
        records[x['relative_path']] = x
    for key, relative in PATHS.items():
        if relative in records:
            rec = records[relative]
            require(len(data[key]) == rec['bytes'] and before[key] == rec['sha256'], 'Existing provenance mismatch: ' + relative)
    require(complete['candidates'] == complete['holm_family_n'] == 20 and complete['method_alpha'] == 0.0125,
            'Frozen family/alpha differs')
    primary = json_bytes(data['primary_json'])
    candidates = json_bytes(data['shortlist'])['selected']
    require(len(candidates) == len(primary) == 20, 'Exactly all twenty candidates required')
    candidate_ids = {x['stable_id'] for x in candidates}
    require(len(candidate_ids) == 20 and candidate_ids == {x['stable_id'] for x in primary}, 'Candidate identities differ')
    holm_check(primary)
    primary_by_gene = {x['gene']: x for x in primary}
    primary_strings = {x['gene']: x for x in read_tsv(data['primary_tsv'])}
    cpu = list(read_tsv(data['cpu_edges']))
    require({x['input_gene'] for x in cpu} == set(primary_by_gene), 'CPU table does not retain all twenty')
    cpu_by_key = {(x['input_gene'], int(x['target_index'])): x for x in cpu}
    require(len(cpu_by_key) == len(cpu), 'Duplicate upstream edge key')
    scores = {}
    for x in read_tsv(data['target_scores']):
        key = (x['candidate_gene'], int(x['target_index']))
        if key in cpu_by_key:
            require(key not in scores, 'Duplicate orthogonal score key')
            scores[key] = x
    require(set(scores) == set(cpu_by_key), 'Missing target score for an existing CPU edge')
    versions = json_bytes(data['runtime_contract'])['family']['runtime_versions']
    baseline = (f"GRNBoost2 / arboreto {versions['arboreto']}; scikit-learn {versions['scikit-learn']}; "
                '1191竞争调控因子;3006目标;seed=0,1,2;同一冻结RNA数据')
    summary, edge_rows = [], []
    for order, candidate in enumerate(candidates, 1):
        gene, stable_id = candidate['feature_id'], candidate['stable_id']
        p, strings = primary_by_gene[gene], primary_strings[gene]
        require(p['stable_id'] == stable_id and p['mode'] == 'PRIMARY_TREE_NORMALIZED', 'Wrong primary candidate/mode')
        for name in ('p_raw', 'p_holm', 'enrichment_over_conditional_expected', 'conditional_expected_overlap', 'jaccard'):
            require((strings[name] == '' and p[name] is None) or float(strings[name]) == p[name], 'JSON/TSV result disagreement: ' + name)
        own = [x for x in cpu if x['input_gene'] == gene]
        nonself = [x for x in own if x['target_stable_id_or_source_motif_id'] != stable_id]
        require(bool(nonself), 'No representative nonself edge; do not invent a weight')
        representative = min(nonself, key=lambda x: (float(x['raw_strict_reference_average_rank']), int(x['target_index'])))
        refs, signed, sign = edge_values(representative)
        algorithm = scores[(gene, int(representative['target_index']))]
        ratio = p['enrichment_over_conditional_expected']
        effect_direction = ('NA_条件期望为零' if ratio is None else '高于条件期望' if ratio > 1
                            else '低于条件期望' if ratio < 1 else '等于条件期望')
        overlaps, overlap_scores = [], []
        for i in p['overlap_target_indices']:
            x = cpu_by_key[(gene, i)]
            require(x['numerically_supported_directional_nonself'] == 'True' and scores[(gene, i)]['primary_algorithm_top50_ties'] == 'True',
                    'Stored overlap detached from actual target membership')
            overlaps.append(gene + '→' + x['target_feature_id'])
            overlap_scores.append(str(edge_values(x)[1]))
        require(len(overlaps) == p['observed_overlap'], 'Overlap count mismatch')
        row = {
            '候选序号_冻结原顺序': order, '候选基因': gene, '候选stable_id': stable_id,
            '检验单位': '候选的完整目标集合_不是单条边', '上游主范围': representative['primary_scope'],
            '上游展示边': gene + '→' + representative['target_feature_id'],
            '上游展示目标stable_id': representative['target_stable_id_or_source_motif_id'],
            '上游展示边原始排名': representative['raw_strict_reference_average_rank'],
            '上游参考导数绝对值中位数': representative['raw_reference_median_abs'],
            '上游参考有符号导数中位数': signed,
            **{f'上游参考导数_seed{s}': refs[s] for s in range(3)},
            '上游导数方向_12均值': sign,
            '上游原数值支持非自边标记': representative['numerically_supported_directional_nonself'],
            '上游展示边同排名数量': sum(float(x['raw_strict_reference_average_rank']) == float(representative['raw_strict_reference_average_rank']) for x in nonself),
            'GRN展示边_树数归一后seed中位数': algorithm['normalized_median'],
            'GRN展示边_官方importance的seed中位数': algorithm['official_raw_median'],
            '展示边属于GRN主top50': algorithm['primary_algorithm_top50_ties'],
            'GRN激活抑制方向': 'NA_非负重要性不识别激活抑制',
            '正交单边P值': 'NA_现有结果未做单边检验',
            '正交候选集合原始P值': strings['p_raw'], '正交候选集合Holm校正P值': strings['p_holm'],
            'Holm校正计算结果_含缺失占位': strings['p_holm_adjustment_result'],
            '缺失P使用保守占位标记': strings['holm_missing_placeholder_used'],
            '观察交集数': p['observed_overlap'], '条件期望交集数': strings['conditional_expected_overlap'],
            '正交效应量_观察除以条件期望': strings['enrichment_over_conditional_expected'],
            '正交效应方向_集合重叠': effect_direction, '正交Jaccard': strings['jaccard'],
            '共同背景目标数': p['common_gene_n'], 'CPU目标集大小': p['cpu_set_n'], 'GRN目标集大小': p['algorithm_set_n'],
            '共同边名称': ';'.join(overlaps) or '无交集',
            '共同边上游参考有符号导数_同顺序': ';'.join(overlap_scores) or 'NA',
            '正交基线方法': baseline, 'GRN主评分公式': ALGORITHM_SCORE,
            '原始P检验': RAW_TEST, 'P值零假设基线': NULL_MODEL,
            '多重检验方法': 'Holm_step_down_FWER', '多重检验校正公式': HOLM,
            '多重检验家族数': 20, '方法alpha': p['method_alpha'],
            '达到正交一致性标准': strings['algorithm_concordance_threshold_met'], '原正交状态': p['status'],
        }
        summary.append(row)
        for x in own:
            grn = scores[(gene, int(x['target_index']))]
            _, edge_signed, edge_sign = edge_values(x)
            edge_rows.append({'candidate_stable_id': stable_id, 'edge_name': gene + '→' + x['target_feature_id'], **x,
                              'reference_signed_median_derived_from_existing_seeds': edge_signed,
                              'derivative_direction_all12_means': edge_sign,
                              **{'grn_' + k: v for k, v in grn.items()},
                              'candidate_set_p_raw_NOT_edge_p': strings['p_raw'],
                              'candidate_set_p_holm_NOT_edge_p': strings['p_holm'],
                              'edge_p_raw': 'NA_NOT_TESTED', 'edge_direction_orthogonally_validated': 'NA_GRN_IMPORTANCE_IS_UNSIGNED'})
    require(len(summary) == 20 and len(edge_rows) == len(cpu), 'Incomplete export')
    require(sum(x['达到正交一致性标准'] == 'True' for x in summary) == complete['method_threshold_met_n'], 'Threshold count changed')
    out.mkdir(parents=True, exist_ok=False)
    write_table(out / 'all20_candidate_details.csv', summary, ',', excel_bom=True)
    write_table(out / 'all20_candidate_details.tsv', summary, '\t')
    write_table(out / 'all_upstream_top50_edges_with_orthogonal_scores.tsv', edge_rows, '\t')
    metadata = {
        'test_unit': 'one candidate and its two complete target sets; not one target edge',
        'baseline': baseline, 'algorithm_score_formula': ALGORITHM_SCORE,
        'raw_p_test': RAW_TEST, 'conditional_null': NULL_MODEL,
        'effect_size': 'observed_overlap / conditional_expected_overlap; >1 above null expectation; <1 below; not activation/repression',
        'holm_formula': HOLM, 'family_n': 20, 'method_alpha': 0.0125,
        'missing_p': 'True P remains missing; explicit adjustment input 1; never label missing true P as 1',
        'representative_edge_rule': REP_RULE,
        'upstream_reference_score': 'Existing raw_reference_median_abs = median_s(abs(raw_mean_seed{s}_route3)); route3=tol1e-9/h1e-4',
        'signed_display_score': 'Median of the three EXISTING signed route3 means; direction checked across all12 stored means',
        'single_edge_p': 'Not present in existing results; NA_NOT_TESTED; candidate P is not assigned as edge P',
        'biological_direction': 'Signed local RNA-velocity sensitivity does not prove binding/activation/repression; GRNBoost2 has no sign',
        'scope': 'DynaVelo CPU full1884/effect_population_all/vx versus GRNBoost2 frozen6035x3006 RNA; three seeds technical, not independent donors',
        'new_model_fits_or_statistical_tests': False,
    }
    with (out / 'methods_and_field_definitions.json').open('x', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    notes = '\n'.join(f'{key}: {value}' for key, value in metadata.items())
    (out / 'READ_ME.txt').write_text(notes + '\n\nCSV保留原始数值精度；控制台仅格式化展示。全部原始CPU top50边不筛除、不补位。\n'
                                     '重新执行请用 --output-dir 指定项目内新的输出目录，脚本拒绝覆盖已有目录。\n'
                                     f'运行示例: python -X utf8 "{Path(__file__).resolve()}" '
                                     '--output-dir results/reports/orthogonal_all20_details_export_NEW\n', encoding='utf-8')
    after = {key: digest(p.read_bytes()) for key, p in paths.items()}
    require(before == after, 'Consumed upstream bytes changed during export; reject this run')
    inventory = [{'relative_path': PATHS[key], 'bytes': len(data[key]), 'sha256_before': before[key],
                  'sha256_after': after[key]} for key in PATHS]
    outputs = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p.read_bytes())}
               for p in sorted(out.iterdir()) if p.is_file()]
    audit = {'status': 'COMPLETE_READ_ONLY_UPSTREAM_UNCHANGED',
             'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'script_sha256': digest(Path(__file__).read_bytes()), 'upstream_inputs': inventory,
             'candidate_rows': 20, 'all_original_upstream_edge_rows': len(edge_rows),
             'stored_holm_formula_recheck_passed': True, 'upstream_content_hashes_unchanged': True,
             'outputs': outputs, 'scientific_claim_added': False}
    with (out / 'export_audit.json').open('x', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print('| 候选 | 上游展示边 | 参考有符号导数中位数 | 原始P | 重叠富集O/E | 相对期望方向 | Holm P |')
    print('|---|---|---:|---:|---:|---|---:|')
    for x in summary:
        fmt = lambda value: 'NA' if value in ('', None) else format(float(value), '.8g')
        print('| ' + ' | '.join([x['候选基因'], x['上游展示边'], fmt(x['上游参考有符号导数中位数']),
                                 fmt(x['正交候选集合原始P值']), fmt(x['正交效应量_观察除以条件期望']),
                                 x['正交效应方向_集合重叠'], fmt(x['正交候选集合Holm校正P值'])]) + ' |')
    print(json.dumps({'status': audit['status'], 'output_dir': str(out), 'candidate_rows': 20,
                      'upstream_edge_rows': len(edge_rows)}, ensure_ascii=False))


if __name__ == '__main__':
    # Windows redirected stdout may otherwise use GBK despite UTF-8 CSV files.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
