from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,gzip,io
import numpy as np
import pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
R=Path(r'D:\dyc_7\DynaVelo_Research');O=R/'results/reports/CEBPA_expression_expansion_20261116_v1';A=O/'analysis'
checks=[]
def ck(name,ok,detail=None):checks.append(dict(check=name,passed=bool(ok),detail=detail));assert ok,(name,detail)
def close(name,a,b,tol=1e-8):
 a=np.asarray(a,float);b=np.asarray(b,float);ck(name,np.allclose(a,b,atol=tol,rtol=1e-8,equal_nan=True),dict(n=int(a.size),max_abs_error=float(np.nanmax(np.abs(a-b))) if a.size and np.isfinite(a-b).any() else 0))
def adjust(p,method):
 p=np.asarray(p,float);n=len(p);ix=np.argsort(p);ans=np.empty(n)
 vals=np.minimum.accumulate((p[ix]*n/np.arange(1,n+1))[::-1])[::-1] if method=='BH' else np.maximum.accumulate(p[ix]*np.arange(n,0,-1))
 ans[ix]=np.minimum(vals,1);return ans
lock=json.loads((O/'BEFORE_FIT_LOCK.json').read_text(encoding='utf-8'));integ=[]
for p,h in lock['expected_sha256'].items():
 observed=hashlib.sha256(Path(p).read_bytes()).hexdigest();ck('locked '+Path(p).name,observed==h);integ.append(dict(path=p,sha256=h,unchanged=observed==h))
(O/'AFTER_FIT_INTEGRITY.json').write_text(json.dumps(integ,indent=2)+'\n',encoding='utf-8')
ends=pd.read_csv(A/'REQUESTED_12_EDGE_ENDPOINTS.csv');allrows=pd.read_csv(A/'ALL_18_GENE_READOUTS.csv');ck('prespecified endpoint counts',len(ends)==12 and len(allrows)==18)
contrasts={};summary=[]
for con in ends.loc[ends.analysis_status=='INDIVIDUAL_SAMPLE_LIMMA','contrast'].unique():
 d=A/con;de=pd.read_csv(d/'all_gene_effects.csv').set_index('symbol');x=pd.read_csv(d/'normalized_expression.tsv',sep='\t').set_index('symbol');sam=pd.read_csv(d/'samples.csv');ko=sam.mutant.to_numpy()==1
 ck(con+' sample alignment',list(x.columns)==sam.column.tolist() and de.index.equals(x.index));ck(con+' independent group unit count',len(sam)==(6 if 'PreGM' in con else 8) and ko.sum()==len(sam)//2)
 xx=x.to_numpy();m0=xx[:,~ko].mean(1);m1=xx[:,ko].mean(1);beta=m1-m0;sigma=(np.sum((xx[:,~ko]-m0[:,None])**2,axis=1)+np.sum((xx[:,ko]-m1[:,None])**2,axis=1))/(len(sam)-2)
 scale=np.sqrt(1/ko.sum()+1/(~ko).sum());se=np.sqrt(de.s2_post)*scale;t=beta/se;p=stats.t.sf(np.abs(t),de.df_total)*2;ci=stats.t.ppf(.975,de.df_total)*se
 close(con+' OLS coefficient',beta,de.log2_effect);close(con+' residual variance',sigma,de.residual_variance);close(con+' unscaled SE',np.repeat(scale,len(de)),de.stdev_unscaled);close(con+' t',t,de.t);close(con+' P',p,de.P);close(con+' BH',adjust(p,'BH'),de.BH_all_genes);close(con+' CI low',beta-ci,de.CI95_low);close(con+' CI high',beta+ci,de.CI95_high)
 rr=allrows[(allrows.contrast==con)];targetcols=['log2_effect','CI95_low','CI95_high','P','BH_all_genes'];close(con+' target effects extracted',rr[targetcols],de.loc[rr.symbol,targetcols])
 g=sam.dataset.iloc[0];mapping=pd.read_csv(O/f'prepared/{g}_probe_mapping.csv',dtype={'probe_id':str});lines=gzip.decompress((O/f'inputs/{g}_series_matrix.txt.gz').read_bytes()).decode('utf-8-sig').splitlines();b=lines.index('!series_matrix_table_begin');e=lines.index('!series_matrix_table_end');raw=pd.read_csv(io.StringIO('\n'.join(lines[b+1:e])),sep='\t',dtype={'ID_REF':str}).set_index('ID_REF')
 for _,r in rr.iterrows():
  probe=mapping.loc[mapping.symbol==r.symbol,'probe_id'];ck(con+' '+r.human_gene+' one source probe',len(probe)==1);original=raw.loc[probe.iloc[0],sam.column].to_numpy();close(con+' '+r.human_gene+' original source values',original,x.loc[r.symbol]);close(con+' '+r.human_gene+' source contrast',np.array([original[ko].mean()-original[~ko].mean()]),np.array([r.log2_effect]));sgn=1 if r.perturbation=='gain' else -1;close(con+' '+r.human_gene+' activation sign',[sgn*r.log2_effect],[r.activation_oriented_effect])
 if g=='GSE146288':ck(con+' unique mouse IDs within arm',sam.mouse_id.nunique()==8)
 c=np.corrcoef(xx.T);off=c[~np.eye(len(sam),dtype=bool)];summary.append(dict(contrast=con,genes=len(de),sample_n=len(sam),correlation_min=float(off.min()),correlation_max=float(off.max())))
thp=pd.read_csv(O/'prepared/GSE119551_exported_group_means.tsv',sep='\t').set_index('Gene')
for _,r in allrows[allrows.dataset=='GSE119551'].iterrows():
 ctx='E' if 'solvent' in r.contrast else 'V';mc=thp.loc[r.human_gene,'avg_NC1_'+ctx];ma=thp.loc[r.human_gene,'avg_CEBPA_'+ctx];ef=np.log2(ma/mc) if ma>0 and mc>0 else np.nan;close('THP1 '+r.human_gene+ctx+' group mean ratio',[ef],[r.log2_effect]);ck('THP1 '+r.human_gene+ctx+' no artificial uncertainty',pd.isna(r.P) and pd.isna(r.CI95_low) and pd.isna(r.SE))
p0=np.where(ends.P.isna(),1,ends.P);close('Holm12 with unassessable endpoints retained',adjust(p0,'holm'),ends.Holm12_P)
sam=pd.read_csv(O/'prepared/GSE146288_samples.csv');ck('HSPC two conditions are same 8 mice',sam.mouse_id.nunique()==8 and (sam.groupby('mouse_id').time.nunique()==2).all())
cov=pd.read_csv(O/'LINCS_MEASUREMENT_COVERAGE.csv').set_index('pr_gene_symbol');ck('Neither target measured landmark in queried L1000 release',(cov.loc[['MPO','CSF3R'],'pr_is_lm']==0).all())
qc=dict(utc=datetime.now(timezone.utc).isoformat(),status='PASS',checks=len(checks),all_checks_passed=all(x['passed'] for x in checks),locked_files=len(integ),modeled_contrasts=summary,scope='Independent Python reconstruction of source target values, group OLS coefficients, residual variance, moderated SE/t/P, all-gene BH and intervals, twelve-endpoint Holm and six descriptive mean ratios. limma posterior variances reused; no independent empirical-Bayes estimation.',scientific_flags=['PreGM knockout-labelled group has higher Cebpa array signal; retains discordant results without relabelling','THP1 only group means, no inferential P/CI','K562 MPO control signal is low relative to its array distribution','HSPC conditions share animals and are not two independent replications'])
(O/'NUMERICAL_QA.json').write_text(json.dumps(qc,indent=2)+'\n',encoding='utf-8');(O/'QA_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8');print(json.dumps(qc,indent=2))
