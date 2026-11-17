from pathlib import Path
import csv, gzip, hashlib, io, json, re, statistics, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
OUT=ROOT/'prepared';assert not OUT.exists(),'Fresh prepared folder required'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table_text(text):return list(csv.DictReader(io.StringIO(text),delimiter='\t'))
def write(path,rows,delimiter=','):
    with path.open('x',encoding='utf-8-sig' if delimiter==',' else 'utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=delimiter);w.writeheader();w.writerows(rows)
prep_path=PROJECT/'results/reports/orthogonal_readout_resume_preparation_20261116_v1/prepared_inputs.json'
prep=json.loads(prep_path.read_text())
short_path=PROJECT/'results/reports/a100_production_20261114/post_screen_v1/locked_scope_shortlist.json'
short=json.loads(short_path.read_text())['selected']
mom_path=PROJECT/prep['target_moments']['relative_path']
assert sha(mom_path)==prep['target_moments']['sha256']
moments=table_text(mom_path.read_text())
edge_path=PROJECT/'results/reports/cpu_locked20_primary_axis_review_20261116_v1/all20_raw_primary_top50_with_flags.tsv'
edges=table_text(edge_path.read_text())
frozen_records={r['relative_path']:r for r in prep['inputs']}
for p in [short_path,edge_path]:assert sha(p)==frozen_records[p.relative_to(PROJECT).as_posix()]['sha256']
input_paths=[prep_path,short_path,mom_path,edge_path]+list((ROOT/'inputs').iterdir())
input_hashes=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in input_paths]
hm=table_text((ROOT/'inputs/human_mouse_orthology.tsv').read_text().replace('[success]',''))
human_to_mouse={};mouse_to_human={};all_human={}
for row in hm:
    h=row['Gene stable ID'];m=row['Mouse gene stable ID']
    all_human.setdefault(h,[]).append(row)
    if m and row['Mouse homology type']=='ortholog_one2one' and row['Mouse orthology confidence [0 low, 1 high]']=='1':
        human_to_mouse.setdefault(h,set()).add(m);mouse_to_human.setdefault(m,set()).add(h)
one2one={h:next(iter(ms)) for h,ms in human_to_mouse.items() if len(ms)==1 and len(mouse_to_human[next(iter(ms))])==1}
mouse_rows=table_text((ROOT/'inputs/mouse_gene_identifiers.tsv').read_text().replace('[success]',''))
entrez_to_mouse={};mouse_symbols={}
for row in mouse_rows:
    m=row['Gene stable ID'];mouse_symbols[m]=row['Gene name']
    e=row['NCBI gene (formerly Entrezgene) ID']
    if e:entrez_to_mouse.setdefault(e,set()).add(m)
annotation=gzip.decompress((ROOT/'inputs/GPL6246.annot.gz').read_bytes()).decode('utf-8-sig')
alines=annotation.splitlines();start=next(i for i,x in enumerate(alines) if x.startswith('ID\t'))
end=next((i for i,x in enumerate(alines[start:],start) if x.startswith('!platform_table_end')),len(alines))
annot=table_text('\n'.join(alines[start:end]))
print('annotation columns',list(annot[0]),flush=True)
genecol=next(k for k in annot[0] if k in ('Gene ID','GeneID','ENTREZ_GENE_ID'))
probe_map={};probe_audit=[]
for r in annot:
    p=r['ID'];ids=set(re.findall(r'\d+',r[genecol]));ms=set().union(*(entrez_to_mouse.get(x,set()) for x in ids)) if ids else set()
    status='UNIQUE_MOUSE_ENSEMBL' if len(ms)==1 and all(len(entrez_to_mouse.get(x,set()))==1 for x in ids) else 'AMBIGUOUS_OR_UNMAPPED'
    if status=='UNIQUE_MOUSE_ENSEMBL':probe_map[p]=next(iter(ms))
    probe_audit.append({'probe_id':p,'NCBI_gene_ids':';'.join(sorted(ids)),'mouse_ensembl_candidates':';'.join(sorted(ms)),'mapping_status':status})
text=gzip.decompress((ROOT/'inputs/GSE52857_series_matrix.txt.gz').read_bytes()).decode()
lines=text.splitlines();begin=lines.index('!series_matrix_table_begin');end=lines.index('!series_matrix_table_end')
reader=csv.reader(lines[begin+1:end],delimiter='\t');header=next(reader);samples_ids=header[1:]
assert len(samples_ids)==12
by_mouse={};probe_count=0
for row in reader:
    probe_count+=1
    if row[0] in probe_map:by_mouse.setdefault(probe_map[row[0]],[]).append(np.array(row[1:],float))
gene_values={m:np.median(np.stack(vals),axis=0) for m,vals in by_mouse.items()}
assert all(np.isfinite(x).all() and (x>0).all() and max(x)<30 for x in gene_values.values())
titles=next(csv.reader([x for x in lines if x.startswith('!Sample_title\t')][0:1],delimiter='\t'))[1:]
sample_rows=[]
for sid,title in zip(samples_ids,titles):
    subtype='CD8alpha_DC' if title.startswith('CD8 DC') else 'CD11b_DC' if title.startswith('CD11b DC') else None
    genotype='WT' if 'wild type' in title else 'KO' if 'Xbp1 KO' in title else None
    assert subtype and genotype
    sample_rows.append({'sample_id':sid,'title':title,'subtype':subtype,'genotype':genotype,'biological_replicate_label':title.rsplit(' ',1)[-1]})
for subtype in ['CD8alpha_DC','CD11b_DC']:
    for genotype in ['WT','KO']:assert sum(x['subtype']==subtype and x['genotype']==genotype for x in sample_rows)==3
candidate_ids={x['stable_id'] for x in short}
coverage=[];mapping={}
for r in moments:
    h=r['stable_id'];m=one2one.get(h,'')
    status='ONE2ONE_MEASURED' if m in gene_values else 'ONE2ONE_NO_ARRAY_COVERAGE' if m else 'NO_UNAMBIGUOUS_HIGH_CONFIDENCE_ONE2ONE'
    info={'human_ensembl':h,'human_symbol':r['symbol'],'mouse_ensembl':m,'mouse_symbol':mouse_symbols.get(m,''),
          'orthology_type':'ortholog_one2one' if m else '', 'orthology_confidence':1 if m else '',
          'mapping_status':status,'mouse_gene_probe_n':len(by_mouse.get(m,[])),
          'candidate_self_target_excluded':h in candidate_ids,
          'included_background':status=='ONE2ONE_MEASURED' and h not in candidate_ids}
    coverage.append(info);mapping[h]=info
background=[r['mouse_ensembl'] for r in coverage if r['included_background']]
assert len(background)==len(set(background)) and len(background)>2000
members=[];coverage20=[]
for c in short:
    g=c['feature_id'];own=[r for r in edges if r['input_gene']==g];assert len(own)==50
    for family in ['qualified','raw']:
        original=[r for r in own if family=='raw' or r['numerically_supported_directional_nonself']=='True']
        rows=[]
        for r in original:
            mp=mapping[r['target_stable_id_or_source_motif_id']]
            rows.append({'candidate':g,'candidate_human_ensembl':c['stable_id'],'family':family,**mp,
                         'included_in_score':mp['included_background'],
                         'original_numeric_support':r['numerically_supported_directional_nonself'],
                         'original_signed_derivative_median3':statistics.median(float(r[f'raw_mean_seed{s}_route3']) for s in range(3)),
                         'original_abs_derivative_score':float(r['raw_reference_median_abs'])})
        n=sum(r['included_in_score'] for r in rows);assert n>=10
        coverage20.append({'candidate':g,'family':family,'original_targets_n':len(rows),
             'one2one_mapped_n':sum(bool(r['mouse_ensembl']) for r in rows),
             'one2one_measured_n':sum(r['mapping_status']=='ONE2ONE_MEASURED' for r in rows),
             'scored_after_common_self_target_exclusion_n':n})
        members.extend(rows)
OUT.mkdir()
write(OUT/'orthology_and_coverage_all3006.csv',coverage)
write(OUT/'probe_mapping_audit.csv',probe_audit)
write(OUT/'candidate_coverage_all20.csv',coverage20)
write(OUT/'candidate_membership.csv',members)
write(OUT/'samples.csv',sample_rows)
with (OUT/'mouse_gene_expression.tsv').open('x',encoding='utf-8',newline='') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['mouse_ensembl']+samples_ids)
    for m,values in sorted(gene_values.items()):w.writerow([m]+values.tolist())
(OUT/'background_mouse_ids.txt').write_text('\n'.join(background)+'\n')
(OUT/'candidate_order.txt').write_text('\n'.join(x['feature_id'] for x in short)+'\n')
result={'status':'PREPARED_NO_DIFFERENTIAL_EXPRESSION_YET','orthology_source':'Ensembl June2026 archive; ortholog_one2one,confidence1,reciprocal unique IDs',
 'annotation_source':'NCBI GEO GPL6246.annot; Entrez Gene -> Ensembl mouse IDs; ambiguous mappings excluded',
 'probe_aggregation':'Median across unambiguously mapped probes, identical for all12 samples, no outcome selection',
 'submitted_normalization':'RMA by original submitter, no second log transformation or re-normalization',
 'array_probe_rows':probe_count,'mapped_mouse_genes':len(gene_values),'common_background_genes':len(background),
 'frozen3006_one2one_n':sum(bool(r['mouse_ensembl']) for r in coverage),'frozen3006_measured_n':sum(r['mapping_status']=='ONE2ONE_MEASURED' for r in coverage),
 'XBP1_target_coverage':[r for r in coverage20 if r['candidate']=='XBP1'],
 'input_hashes':input_hashes,'prepared_files':[{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir())]}
with (ROOT/'preparation_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','prepared_files')}),flush=True)
