from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import gzip,io,hashlib,json,re,sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
R=Path(r'D:\dyc_7\DynaVelo_Research');O=R/'results/reports/CEBPA_expression_expansion_20261116_v1';I=O/'inputs';P=O/'prepared';P.mkdir(exist_ok=True)
assert not (O/'BEFORE_FIT_LOCK.json').exists()
def parse_soft(path):
 rec=[];cur=None
 for ln in path.read_text(encoding='utf-8').splitlines():
  if ln.startswith('^SAMPLE = '):cur={'GSM':ln.split(' = ',1)[1],'characteristics':[]};rec.append(cur)
  elif cur is not None and ln.startswith('!Sample_title = '):cur['title']=ln.split(' = ',1)[1]
  elif cur is not None and ln.startswith('!Sample_characteristics_ch1 = '):cur['characteristics'].append(ln.split(' = ',1)[1])
 for r in rec:r['characteristics']=' | '.join(r['characteristics'])
 return pd.DataFrame(rec)
def matrix(name):
 lines=gzip.decompress((I/f'{name}_series_matrix.txt.gz').read_bytes()).decode('utf-8-sig').splitlines();b=lines.index('!series_matrix_table_begin');e=lines.index('!series_matrix_table_end');d=pd.read_csv(io.StringIO('\n'.join(lines[b+1:e])),sep='\t',dtype={'ID_REF':str}).set_index('ID_REF')
 assert not d.index.duplicated().any() and np.isfinite(d.to_numpy()).all();return d
mp=pd.read_csv(I/'probe_mapping_audit.csv',dtype=str).fillna('');mp=mp[mp.mapping_status=='UNIQUE_MOUSE_ENSEMBL'].set_index('probe_id');assert not mp.index.duplicated().any()
mouse=pd.read_csv(I/'mouse_gene_identifiers.tsv',sep='\t',dtype=str).fillna('');ms=defaultdict(set)
for gene,symbol in zip(mouse.iloc[:,0],mouse.iloc[:,1]):
 if symbol:ms[gene].add(symbol)
mouse_uniq={gene:next(iter(sym)) for gene,sym in ms.items() if len(sym)==1}
plines=gzip.decompress((I/'GPL17077.soft.gz').read_bytes()).decode('utf-8').splitlines();b=plines.index('!platform_table_begin');e=plines.index('!platform_table_end');platform=pd.read_csv(io.StringIO('\n'.join(plines[b+1:e])),sep='\t',dtype=str,keep_default_na=False).set_index('ID');assert not platform.index.duplicated().any()
summary=[];all_samples=[]
for g in ['GSE58743','GSE84388','GSE146288']:
 x=matrix(g);sam=parse_soft(O/f'metadata/{g}_metadata_only.soft').set_index('GSM').loc[list(x.columns)].rename_axis('GSM').reset_index();assert len(sam)==x.shape[1]
 audit=pd.DataFrame(index=x.index);audit.index.name='probe_id'
 if g=='GSE58743':
  ann=platform.reindex(x.index).fillna('');sym=ann.GENE_SYMBOL;ok=ann.CONTROL_TYPE.str.upper().isin(['0','FALSE'])&sym.str.match(r'^[A-Za-z][A-Za-z0-9_.-]*$');audit['symbol']=sym.where(ok,'');audit['source_ensembl']=ann.ENSEMBL_ID;audit['mapping_status']=np.where(ok,'UNIQUE_PLATFORM_GENE_SYMBOL','AMBIGUOUS_UNMAPPED_OR_CONTROL')
  sam['mutant']=sam.title.str.contains('CEBPA').astype(int);sam['condition']=np.where(sam.mutant.eq(1),'CEBPA_induced','CTR_induced');sam['mouse_id']='';sam['time']='48h';sam['unit']='reported independent biological culture repeat; stable K562 construct';sam['perturbation']='gain';sam['replicate']=sam.title.str.extract(r'replicate(\d+)')[0]
 else:
  ann=mp.reindex(x.index).fillna('');ens=ann.mouse_ensembl_candidates;sym=ens.map(mouse_uniq).fillna('');ok=sym.ne('');audit['symbol']=sym.where(ok,'');audit['source_ensembl']=ens;audit['mapping_status']=np.where(ok,'UNIQUE_MOUSE_ID_AND_SYMBOL','AMBIGUOUS_OR_UNMAPPED')
  sam['perturbation']='loss'
  if g=='GSE84388':
   sam['mutant']=sam.title.str.contains('Mx1Cre').astype(int);sam['condition']=np.where(sam.mutant.eq(1),'KO','WT');sam['mouse_id']='GEO_individual_'+sam.GSM;sam['time']='18d_after_pIC';sam['unit']='individual female mouse; sorted PreGM';sam['replicate']=sam.title.str.extract(r'rep(\d+)')[0]
  else:
   sam['mutant']=sam.title.str.startswith('ko_').astype(int);sam['time']=np.where(sam.title.str.contains('_4h'),'FLT3L4h','untreated');sam['condition']=np.where(sam.mutant.eq(1),'KO','WT');sam['mouse_id']=sam.title.str.extract(r'_(\d{3})(?:_2)?:')[0];assert sam.mouse_id.notna().all() and sam.mouse_id.nunique()==8;sam['unit']='individual mouse; same ID repeated across treatment arms';sam['replicate']=sam.title.str.extract(r'[Rr]ep\s*(\d+)')[0]
 sam['dataset']=g;sam['column']=sam.GSM;sam['pairing']='NONE_BETWEEN_GENOTYPES_OR_CONSTRUCTS';sam['batch']='UNREPORTED';all_samples.append(sam)
 valid=audit.symbol.ne('');mapped=x.loc[valid].copy();mapped['symbol']=audit.loc[valid,'symbol'];gene=mapped.groupby('symbol',sort=True).median();assert not gene.index.duplicated().any()
 for target in ['CEBPA','MPO','CSF3R']:
  lookup=target if g=='GSE58743' else {'CEBPA':'Cebpa','MPO':'Mpo','CSF3R':'Csf3r'}[target]
  assert lookup in gene.index,(g,target,'Target absent; stop before fit')
 gene.index.name='symbol';gene.to_csv(P/f'{g}_gene_expression.tsv',sep='\t');sam.to_csv(P/f'{g}_samples.csv',index=False);audit.to_csv(P/f'{g}_probe_mapping.csv')
 qc=pd.DataFrame(dict(probes=x.index,pooled_mean=x.mean(axis=1),pooled_min=x.min(axis=1),pooled_max=x.max(axis=1)));qc.to_csv(P/f'{g}_pooled_probe_QC.csv',index=False)
 summary.append(dict(dataset=g,source_probes=len(x),valid_probes=int(valid.sum()),mapped_genes=len(gene),samples=len(sam),gene_collapse='per-sample median of all unique annotated probes',target_probe_coverage={t:int(audit.symbol.eq(t if g=='GSE58743' else {'CEBPA':'Cebpa','MPO':'Mpo','CSF3R':'Csf3r'}[t]).sum()) for t in ['CEBPA','MPO','CSF3R']}))
thp=pd.read_csv(I/'GSE119551_abundance.csv.gz',sep=';',decimal=',');assert list(thp.columns)==['Gene','avg_CEBPA_E','avg_CEBPA_V','avg_NC1_E','avg_NC1_V']
assert np.isfinite(thp.iloc[:,1:].to_numpy(float)).all() and (thp.iloc[:,1:].to_numpy(float)>=0).all()
thp.to_csv(P/'GSE119551_exported_group_means.tsv',sep='\t',index=False)
parse_soft(O/'metadata/GSE119551_samples.soft').to_csv(P/'GSE119551_sample_metadata_unusable_for_variance.csv',index=False)
pd.concat(all_samples,ignore_index=True).to_csv(P/'all_individual_samples.csv',index=False)
(O/'PREPARATION_QA.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),arrays=summary,THP1=dict(rows=len(thp),group_mean_columns=4,individual_expression_available=False,duplicate_gene_symbols=int(thp.Gene.duplicated().sum())),effects_not_yet_calculated=True),indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2));print('THP1 group means only; no inferential fit')
