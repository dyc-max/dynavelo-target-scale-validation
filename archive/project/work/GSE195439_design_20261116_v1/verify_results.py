from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import pandas as pd
import numpy as np
from scipy.stats import rankdata,t,spearmanr
sys.stdout.reconfigure(encoding='utf-8')
root=Path(r'D:\dyc_7\DynaVelo_Research')
out=root/'results/reports/GSE195439_xbp1_replication_20261116_v1'
res=out/'outputs'
old=root/'results/reports/xbp1_GSE52857_external_validation_20261116_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bh(p):
 order=np.argsort(p,kind='stable');m=len(p);q=np.empty(m)
 q[order]=np.minimum(1,np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]);return q
run=json.loads((out/'RUN_COMPLETE.json').read_text())
assert run['status']=='COMPLETE' and run['exit_code']==0 and run['process_exited']
pins=json.loads((out/'RUN_STARTED.json').read_text())['expected_sha256']
for path,h in pins.items():assert sha(Path(path))==h,path
summary=pd.read_csv(res/'all20_candidate_summary.csv');perms=pd.read_csv(res/'global_contrast_all_permutations.csv')
members=pd.read_csv(out/'prepared/candidate_membership.csv',keep_default_na=False)
gene_tables=[];checks=[]
for variant in ['matched_WT3_KO4','all_WT4_KO4_sensitivity']:
 folder=res/variant;de=pd.read_csv(folder/'all_gene_DE.csv').set_index('source_symbol')
 E=pd.read_csv(folder/'voom_logCPM.tsv',sep='\t',index_col=0)
 W=pd.read_csv(folder/'voom_weights.tsv',sep='\t',index_col=0).to_numpy()
 qc=pd.read_csv(folder/'sample_QC.csv');ko=qc['group'].to_numpy()=='KO'
 assert list(de.index)==list(E.index) and list(E.columns)==list(qc.title) and W.shape==E.shape
 independent_fc=np.average(E.to_numpy()[:,ko],axis=1,weights=W[:,ko])-np.average(E.to_numpy()[:,~ko],axis=1,weights=W[:,~ko])
 fc_err=float(np.max(np.abs(independent_fc-de.log2FC.to_numpy())))
 p_recomputed=2*t.sf(np.abs(de.moderated_t.to_numpy()),de.df_total.to_numpy())
 p_err=float(np.max(np.abs(p_recomputed-de.P.to_numpy())))
 bh_err=float(np.max(np.abs(bh(de.P.to_numpy())-de.BH.to_numpy())))
 ci_err=float(np.max(np.abs(de.log2FC-t.ppf(.975,de.df_total)*de.moderated_SE-de.CI95_low)))
 # R qt and SciPy ppf differ by up to1e-9 at a few df values; keep a
 # separate1e-8 CI tolerance rather than relaxing coefficient/P checks.
 assert max(fc_err,p_err,bh_err)<1e-10 and ci_err<1e-8
 bg=pd.read_csv(folder/'observed_background_scores.csv');indices=E.index.get_indexer(bg.source_symbol)
 rr=(rankdata(np.abs(de.loc[bg.source_symbol,'moderated_t']))-.5)/len(bg)
 ctrl=E.to_numpy()[indices][:,~ko];mu=ctrl.mean(axis=1);variance=ctrl.var(axis=1,ddof=1)
 bins=lambda a:np.searchsorted(np.unique(np.quantile(a,[.2,.4,.6,.8])),a,side='right')
 strata=5*bins(mu)+bins(variance);adjusted=rr.copy()
 for s in np.unique(strata):adjusted[strata==s]-=rr[strata==s].mean()
 assert np.array_equal(strata,bg.stratum.to_numpy())
 score_err=float(np.max(np.abs(adjusted-bg.adjusted_score.to_numpy())))
 assert score_err<1e-12
 allocations=pd.read_csv(folder/'allocations.csv')
 assert len(allocations)==(35 if variant=='matched_WT3_KO4' else 70)
 assert allocations.KO_samples.nunique()==len(allocations) and allocations.is_observed.sum()==1
 pcs=pd.read_csv(folder/'all_candidate_permutation_scores.csv')
 for family in ['qualified','raw']:
  ss=summary[(summary.variant==variant)&(summary.family==family)].copy()
  assert len(ss)==20
  assert np.max(np.abs(rankdata(-ss.adjusted_score.to_numpy())-ss['rank'].to_numpy()))<1e-12
  for _,row in ss.iterrows():
   mm=members[(members.candidate==row.candidate)&(members.family==family)]
   include=mm.eligible_before_expression_filter.astype(str).str.lower().eq('true')&mm.source_symbol.isin(bg.source_symbol)
   set_symbols=mm.loc[include,'source_symbol']
   assert len(set_symbols)==row.scored_n
   score=bg.set_index('source_symbol').loc[set_symbols,'adjusted_score'].mean()
   assert abs(score-row.adjusted_score)<1e-12
  own=ss[ss.candidate=='XBP1'].iloc[0]
  gap=own.adjusted_score-ss.loc[ss.candidate!='XBP1','adjusted_score'].mean()
  pp=perms[(perms.variant==variant)&(perms.family==family)]
  assert abs(gap-own.XBP1_minus_mean_other19)<1e-12
  cols=[family+'__'+g for g in ss.candidate if g!='XBP1']
  delta=pcs[family+'__XBP1']-pcs[cols].mean(axis=1)
  assert np.max(np.abs(delta.to_numpy()-pp.global_gap.to_numpy()))<1e-12
  p=float(np.mean(delta>=gap-1e-12))
  assert abs(p-own.global_exact_P)<1e-12
  assert float(rankdata(-ss.adjusted_score.to_numpy())[list(ss.candidate).index('XBP1')])==own['rank']
 mm=members[(members.candidate=='XBP1')&(members.family=='qualified')].copy()
 merged=mm.merge(de.reset_index(),on='source_symbol',how='left',validate='many_to_one')
 merged['variant']=variant
 merged['scored_after_filter']=merged.eligible_before_expression_filter.astype(str).str.lower().eq('true')&merged.source_symbol.isin(bg.source_symbol)
 merged['final_coverage_status']=np.where(merged.mapping_status_RNAseq!='ELIGIBLE_BEFORE_COUNT_FILTER',merged.mapping_status_RNAseq,np.where(merged.scored_after_filter,'SCORED','LOW_COUNT_FILTERED'))
 gene_tables.append(merged)
 corr=pd.read_csv(folder/'sample_correlation.csv',index_col=0).to_numpy();corr=corr[np.triu_indices(len(corr),1)]
 checks.append(dict(variant=variant,genes=len(de),background=len(bg),max_weighted_OLS_logFC_error=fc_err,
  max_t_P_error=p_err,max_BH_error=bh_err,max_CI_error=ci_err,max_score_error=score_err,
  raw_library_counts_min=int(qc.raw_library_counts.min()),raw_library_counts_max=int(qc.raw_library_counts.max()),
  sample_pair_correlation_min=float(corr.min()),sample_pair_correlation_max=float(corr.max()),
  TMM_min=float(qc.TMM_factor.min()),TMM_max=float(qc.TMM_factor.max()),total_gene_BH005=int((de.BH<.05).sum())))
targets=pd.concat(gene_tables,ignore_index=True)
targets.to_csv(res/'XBP1_all48_target_details.csv',index=False)
primary=targets[targets.variant=='matched_WT3_KO4'].copy()
previous=pd.read_csv(old/'outputs/XBP1_all_frozen_target_details.csv')
cross=[];cross_summary=[]
for subtype in ['CD8alpha_DC','CD11b_DC']:
 p=previous[(previous.family=='qualified')&(previous.subtype==subtype)][['human_ensembl','log2FC_KO_minus_WT','FDR_BH_all_measured_genes','included_in_score']]
 joined=primary.merge(p,on='human_ensembl',how='left',validate='one_to_one')
 joined['previous_subtype']=subtype
 included=joined.scored_after_filter&joined.included_in_score.astype(str).str.lower().eq('true')
 shared=joined[included & joined.log2FC.notna() & joined.log2FC_KO_minus_WT.notna()].copy()
 rho=spearmanr(shared.log2FC,shared.log2FC_KO_minus_WT).statistic
 same=(np.sign(shared.log2FC)==np.sign(shared.log2FC_KO_minus_WT))
 cross_summary.append({'previous_subtype':subtype,'common_scored_targets_n':len(shared),'same_sign_n':int(same.sum()),
   'logFC_spearman_descriptive':float(rho),'both_gene_FDR005_n':int(((shared.BH<.05)&(shared.FDR_BH_all_measured_genes<.05)).sum())})
 cross.append(joined)
pd.concat(cross,ignore_index=True).to_csv(res/'all48_cross_study_gene_details.csv',index=False)
pd.DataFrame(cross_summary).to_csv(res/'cross_study_descriptive_summary.csv',index=False)
prior_seven=pd.concat(cross,ignore_index=True)
prior_seven=prior_seven[(prior_seven.previous_subtype=='CD8alpha_DC')&(prior_seven.FDR_BH_all_measured_genes<.05)]
assert len(prior_seven)==7
prior_seven.to_csv(res/'previous_seven_responder_followup.csv',index=False)
loo=pd.read_csv(res/'leave_one_pool_out.csv');qloo=loo[loo.family=='qualified']
assert len(qloo)==7 and qloo.omitted_sample.nunique()==7
own=summary[(summary.family=='qualified')&(summary.candidate=='XBP1')]
main=own[own.variant=='matched_WT3_KO4'].iloc[0];alt=own[own.variant=='all_WT4_KO4_sensitivity'].iloc[0]
rule={'primary_positive_gap':bool(main.XBP1_minus_mean_other19>0),'primary_rank_le3':bool(main['rank']<=3),
 'all7_LOO_positive_gap':bool((qloo.global_gap>0).all()),'all_WT_sensitivity_positive_gap':bool(alt.XBP1_minus_mean_other19>0)}
audit={'status':'PASS_NUMERICAL_AND_FILE_CHECKS','utc':datetime.now(timezone.utc).isoformat(),'pinned_files_unchanged_n':len(pins),
 'scope':'Same-agent independent arithmetic and pinned file checks; no external blind review, no complete second R fit',
 'variant_checks':checks,'all80_candidate_scores_and_ranks_verified':True,'exact_global_P_values_verified':4,
 'primary_XBP1':main.to_dict(),'sensitivity_XBP1':alt.to_dict(),'LOO_positive_n':int((qloo.global_gap>0).sum()),
 'LOO_gap_min':float(qloo.global_gap.min()),'LOO_gap_max':float(qloo.global_gap.max()),'LOO_rank_min':float(qloo.own_rank.min()),'LOO_rank_max':float(qloo.own_rank.max()),
 'cross_study':cross_summary,'resource_priority_rule':rule,'continue_targeted_exploration':all(rule.values()),
 'confirmation':'NOT_CONFIRMED','new_gene_level_findings':primary[primary.scored_after_filter&(primary.BH<.05)][['human_symbol','source_symbol','log2FC','CI95_low','CI95_high','BH']].to_dict('records'),
 'source_exposure':'GSE195439 now analyzed; not available as untouched future confirmation','frozen_original_E1_and_readout_unchanged':True}
(out/'FINAL_QA.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
print('PRIOR7');print(prior_seven[['human_symbol','log2FC_KO_minus_WT','log2FC','BH','final_coverage_status']].to_string(index=False))
