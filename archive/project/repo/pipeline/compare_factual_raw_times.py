"""Descriptive paired seed0/1 raw Beta moments; no p values or model execution."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import audit_factual_run_readout as q
REFERENCE='results/reports/epoch90_terminal_posterior_audit_20261112.json'
REFERENCE_SHA='9cc265a708935ed7e7c5381b143d376ea797990e28645f1e7cecf050fda15cea'
def coefficient(a,b,*,rank=False):
 import numpy as np
 from scipy.stats import rankdata
 if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
 x,y=(rankdata(a,method='average'),rankdata(b,method='average')) if rank else (a,b)
 return float(np.corrcoef(x,y)[0,1])
def describe(x):
 import numpy as np
 return {'n':len(x),'min':float(np.min(x)),'q25':float(np.quantile(x,.25)),'median':float(np.median(x)),'q75':float(np.quantile(x,.75)),'max':float(np.max(x)),'IQR':float(np.quantile(x,.75)-np.quantile(x,.25))}
def compare_arrays(cell0,cell1,labels0,labels1,mean0,sd0,mean1,sd1):
 import numpy as np
 if list(cell0)!=list(cell1) or len(set(cell0))!=len(cell0) or not np.array_equal(labels0,labels1):raise ValueError('Exact paired cell/label order required; no automatic alignment')
 q.validate_times(mean0,sd0);q.validate_times(mean1,sd1);correlations=[];distributions=[];directions=[]
 for kind,name,mask in q.selections(labels0):
  if not mask.any():continue
  for field,a,b in [('raw_beta_mean',mean0,mean1),('analytic_beta_sd',sd0,sd1)]:
   correlations.append({'group_type':kind,'group':name,'field':field,'n_cells':int(mask.sum()),'Pearson':coefficient(a[mask],b[mask]),'Spearman_average_ties':coefficient(a[mask],b[mask],rank=True),'coefficient_scope':'descriptive same cells; seeds/cells are not biological replicates'})
   for seed,x in [(0,a[mask]),(1,b[mask])]:distributions.append({'group_type':kind,'group':name,'seed':seed,'field':field,**describe(x)})
 for seed,t in [(0,mean0),(1,mean1)]:
  for source,target in q.PAIRS:
   a=t[labels0==source];b=np.sort(t[labels0==target]);row={'seed':seed,'source_label':source,'target_label':target,'n_source':len(a),'n_target':len(b),'median_delta_target_minus_source':None,'fraction_expected_order_half_ties':None,'median_order':'UNDEFINED_ABSENT_LABEL'}
   if len(a) and len(b):
    lo=np.searchsorted(b,a,side='left');hi=np.searchsorted(b,a,side='right');delta=float(np.median(b)-np.median(a))
    row.update(median_delta_target_minus_source=delta,fraction_expected_order_half_ties=float(np.mean((len(b)-hi+.5*(hi-lo))/len(b))),median_order='expected' if delta>0 else 'reversed' if delta<0 else 'tied')
   directions.append(row)
 return {'correlations':correlations,'distributions':distributions,'all_eight_priors_each_seed':directions,'P_values_generated':False,'scientific_pass':False,'semantics':'Raw analytic Beta mean/SD only. Spearman uses average ties, not saved ordinal expansion. No elapsed-time identification, calibrated uncertainty, causal validation or population P values.'}
def execute(root,review_path,review_sha):
 import numpy as np,h5py
 from anndata.experimental import read_elem
 raw=Path(review_path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=review_sha:raise ValueError('Comparison review SHA differs')
 review=json.loads(raw)
 if review.get('schema')!='dynavelo_time_comparison_review_v1' or review.get('execution_authorized') is not True or review.get('reviewer')!='/root' or review.get('script_sha256')!=q.sha(__file__) or review.get('readout_script_sha256')!=q.sha(q.__file__) or review.get('scientific_pass') is not False:raise PermissionError('Root-reviewed paired descriptive comparison required')
 new_path=q.checked(root,review['new_audit']);new=q.read(new_path)
 if new.get('schema')!='dynavelo_factual_readout_v1' or new.get('status')!='DESCRIPTIVE_INTERNAL_READOUT_NO_SCIENTIFIC_PASS' or new.get('seed')!=1 or new.get('run_id')!=review.get('run_id') or new.get('script_sha256')!=q.sha(q.__file__) or new.get('scientific_pass') is not False:raise ValueError('Completed new seed1 readout required')
 completion=q.read(new_path.parent/'COMPLETE.json')
 if completion.get('audit')!=q.record(new_path,root):raise ValueError('Readout COMPLETE differs')
 for rec in new['artifacts']:q.checked(root,rec)
 time_rec=next(r for r in new['artifacts'] if r['relative_path'].endswith('/cell_raw_beta_time.tsv'));time_path=q.checked(root,time_rec)
 with time_path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
 ids1=[r['cell_id'] for r in rows];labels1=np.asarray([r['author_state'] for r in rows]);split1=[r['split'] for r in rows];m1=np.asarray([float(r['raw_beta_mean']) for r in rows]);s1=np.asarray([float(r['analytic_beta_sd']) for r in rows])
 reference_path=q.safe(root,REFERENCE)
 if q.sha(reference_path)!=REFERENCE_SHA:raise ValueError('Previously audited seed0 reference drift')
 reference=q.read(reference_path);run0=reference['run_id'];pred_rel='data/processed/GSE284047/8489-MV-1/model_runs/'+run0+'/adata_rna_pred.h5ad';records={r['source_relative_path']:r for r in reference['source_manifest_records']}
 pred=q.checked(root,{'relative_path':pred_rel,'bytes':records[pred_rel]['bytes'],'sha256':records[pred_rel]['sha256']})
 metrics_rel='checkpoints/GSE284047/'+run0+'/metrics.json';metrics_path=q.checked(root,{'relative_path':metrics_rel,'bytes':records[metrics_rel]['bytes'],'sha256':records[metrics_rel]['sha256']});old_metrics=q.read(metrics_path)
 with h5py.File(pred,'r') as f:obs=read_elem(f['obs'])
 if len(rows)!=6035 or obs['celltype'].value_counts().to_dict()!=q.STATE_COUNTS or obs['split'].astype(str).tolist()!=split1 or split1.count('train')!=4828 or split1.count('validation')!=1207:raise ValueError('Same full context/split required')
 result=compare_arrays(obs.index.tolist(),ids1,obs['celltype'].astype(str).to_numpy(),labels1,obs['latent_time_mean'].to_numpy(float),obs['latent_time_std'].to_numpy(float),m1,s1)
 if not q.RUN_RE.fullmatch(review.get('run_id','')) or not __import__('re').fullmatch('[A-Za-z0-9_-]{8,100}',review.get('output_id','')):raise ValueError('Bound output identity required')
 if q.sha(review_path)!=review_sha or q.sha(new_path)!=review['new_audit']['sha256'] or q.sha(pred)!=records[pred_rel]['sha256'] or q.sha(time_path)!=time_rec['sha256'] or q.sha(__file__)!=review['script_sha256']:raise ValueError('Source/review changed during comparison')
 out=q.safe(root,'results/reports/factual_time_comparisons/'+review['output_id']);out.mkdir(parents=True,exist_ok=False)
 for key in ['correlations','distributions','all_eight_priors_each_seed']:q.write_table(out/(key+'.tsv'),result[key])
 result.update(schema='dynavelo_paired_raw_beta_comparison_v1',status='DESCRIPTIVE_SEED_STABILITY_ONLY',runs={'seed0':run0,'seed1':new['run_id']},best_epochs={'seed0':old_metrics['best_validation_loss_epoch'],'seed1':new['best_epoch']},cell_axis_sha256=q.digest(ids1),n_cells=6035,new_audit=review['new_audit'],reference_audit_sha256=REFERENCE_SHA,script_sha256=q.sha(__file__),readout_script_sha256=q.sha(q.__file__),review_sha256=review_sha)
 q.save(out/'comparison.json',result);q.save(out/'COMPLETE.json',{'comparison':q.record(out/'comparison.json',root),'scientific_pass':False});return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path);p.add_argument('--review-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps({'status':'INSPECT_ONLY_NO_RESULTS_READ','fields':['raw_beta_mean','analytic_beta_sd'],'group_count_per_seed':16,'direction_priors_per_seed':8,'P_values_generated':False,'scientific_pass':False}));return
 if a.review is None or not a.review_sha256 or not a.review.absolute().is_relative_to(q.ROOT):raise ValueError('Project review path/SHA required')
 r=execute(q.ROOT,a.review.absolute(),a.review_sha256);print(json.dumps({'status':r['status'],'runs':r['runs']}))
if __name__=='__main__':main()
