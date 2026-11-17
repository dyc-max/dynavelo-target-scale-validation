from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,sys
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
root=Path(r'D:\dyc_7\DynaVelo_Research')
out=root/'results/reports/GSE195439_xbp1_replication_20261116_v1'
old=root/'results/reports/xbp1_GSE52857_external_validation_20261116_v1'
prep=out/'prepared';inputs=out/'inputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
initial=json.loads((out/'PLAN_LOCK.json').read_text())
manifest=initial['expected_sha256']
for p,expected in manifest.items():assert sha(Path(p))==expected
schema=out/'INPUT_SCHEMA_AMENDMENT_01.md'
identifiers=old/'inputs/mouse_gene_identifiers.tsv'
manifest.update({str(schema):sha(schema),str(identifiers):sha(identifiers)})
(out/'PLAN_SCHEMA_LOCK_02.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'stage':'BEFORE_EFFECTS_AFTER_SYMBOL_SCHEMA_INSPECTION','expected_sha256':manifest},indent=2)+'\n',encoding='utf-8')
(out/'preparation_failure_v1.json').write_text(json.dumps({'stage':'count ID parsing','reason':'Source index is unique gene symbols; erroneous Ensembl version-strip created false duplicates','effects_computed':False,'resolution':'Exact source symbols and official unique mouse ID/name mapping; see INPUT_SCHEMA_AMENDMENT_01.md'},indent=2)+'\n',encoding='utf-8')
counts=pd.read_csv(inputs/'GSE195439_Bulk_RNA_seq_raw_counts.txt.gz',sep='\t',index_col=0)
assert counts.index.is_unique
samples=pd.read_csv(prep/'samples.csv')
assert set(counts.columns)==set(samples.title)
bookkeeping=counts.index.astype(str).str.startswith('__')
counts.loc[bookkeeping].to_csv(prep/'excluded_bookkeeping_rows.csv')
counts=counts.loc[~bookkeeping,samples.title]
v=counts.to_numpy(dtype=float)
assert np.isfinite(v).all() and (v>=0).all() and np.all(v==np.floor(v))
counts.to_csv(prep/'raw_counts.tsv',sep='\t',index_label='source_symbol')
axis=pd.read_csv(old/'prepared/orthology_and_coverage_all3006.csv',keep_default_na=False)
members=pd.read_csv(old/'prepared/candidate_membership.csv',keep_default_na=False)
orth=pd.read_csv(old/'inputs/human_mouse_orthology.tsv',sep='\t',keep_default_na=False)
valid=orth[(orth['Mouse homology type']=='ortholog_one2one')&(orth['Mouse orthology confidence [0 low, 1 high]'].astype(str)=='1')].copy()
valid=valid.drop_duplicates(['Gene stable ID','Mouse gene stable ID'])
valid=valid[~valid['Gene stable ID'].duplicated(False)&~valid['Mouse gene stable ID'].duplicated(False)]
maps=valid.set_index('Gene stable ID')['Mouse gene stable ID'].to_dict()
ids=pd.read_csv(identifiers,sep='\t',keep_default_na=False)[['Gene stable ID','Gene name']].drop_duplicates()
unique=ids[(ids['Gene name']!='')&~ids['Gene stable ID'].duplicated(False)&~ids['Gene name'].duplicated(False)]
id_symbol=unique.set_index('Gene stable ID')['Gene name'].to_dict()
symbol_id={v:k for k,v in id_symbol.items()}
candidate_ids=set(members.candidate_human_ensembl)
excluded_mouse={maps[x] for x in candidate_ids if x in maps}
for frame in [axis,members]:
 frame['mouse_ensembl']=frame['human_ensembl'].map(maps).fillna('')
 frame['source_symbol']=frame['mouse_ensembl'].map(id_symbol).fillna('')
 frame['RNAseq_row_present']=frame['source_symbol'].isin(counts.index)
 frame['self_column_excluded']=frame['mouse_ensembl'].isin(excluded_mouse)
 frame['eligible_before_expression_filter']=frame.RNAseq_row_present&~frame.self_column_excluded
 frame['mapping_status_RNAseq']=np.where(frame.mouse_ensembl=='','NO_HIGH_CONFIDENCE_RECIPROCAL_ONE2ONE',np.where(frame.source_symbol=='','NO_UNIQUE_OFFICIAL_MOUSE_SYMBOL',np.where(~frame.RNAseq_row_present,'SOURCE_SYMBOL_NOT_PRESENT',np.where(frame.self_column_excluded,'CANDIDATE_SELF_COLUMN_EXCLUDED','ELIGIBLE_BEFORE_COUNT_FILTER'))))
 frame.drop(columns=[c for c in ['mapping_status','mouse_gene_probe_n','included_background','included_in_score','candidate_self_target_excluded'] if c in frame],inplace=True)
axis.to_csv(prep/'target_axis_orthology.csv',index=False)
members.to_csv(prep/'candidate_membership.csv',index=False)
pd.DataFrame({'source_symbol':counts.index,'mouse_ensembl':[symbol_id.get(x,'') for x in counts.index]}).to_csv(prep/'source_gene_annotation.csv',index=False)
(prep/'candidate_order.txt').write_bytes((old/'prepared/candidate_order.txt').read_bytes())
assert len(axis)==3006 and len(members)==1952
for p,expected in manifest.items():assert sha(Path(p))==expected
coverage=members.groupby(['candidate','family']).agg(original_n=('human_ensembl','size'),mapped_in_source_n=('eligible_before_expression_filter','sum')).reset_index()
coverage.to_csv(prep/'candidate_coverage_before_filter.csv',index=False)
summary={'status':'PREPARED_BEFORE_EFFECTS','utc':datetime.now(timezone.utc).isoformat(),'raw_mouse_gene_symbol_rows':len(counts),
 'expected_sha256':{**manifest,**{str(p):sha(p) for p in prep.iterdir() if p.is_file()},str(inputs/'GSE195439_Bulk_RNA_seq_raw_counts.txt.gz'):sha(inputs/'GSE195439_Bulk_RNA_seq_raw_counts.txt.gz')},
 'primary_n_WT':3,'primary_n_KO':4,'primary_allocations':35,'primary_min_exact_P':1/35,'confirmation_alpha_upper_bound':.0125,
 'confirmation_status':'NOT_ATTAINABLE_BY_SPECIFIED_EXACT_TEST','scientific_status':'EXPLORATORY_ONLY'}
(out/'preparation_receipt.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='expected_sha256'},ensure_ascii=False))
print(coverage[coverage.candidate=='XBP1'].to_string(index=False))
