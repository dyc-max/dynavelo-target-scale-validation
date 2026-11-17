args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1],winslash="/")
out <- file.path(root,"outputs")
if(file.exists(file.path(out,"COMPLETE.txt"))) stop("Refuse overwrite")
dir.create(out,showWarnings=FALSE)
.libPaths(c(file.path(root,"R_library"),"D:/dyc_7/DynaVelo_Research/results/reports/xbp1_GSE52857_external_validation_20261116_v1/R_library",.libPaths()))
library(edgeR)
library(limma)
stopifnot(as.character(packageVersion("limma"))=="3.68.5",as.character(packageVersion("edgeR"))=="4.10.7")
y <- as.matrix(read.delim(file.path(root,"prepared/raw_counts.tsv"),row.names=1,check.names=FALSE))
storage.mode(y)<-"double"
samples<-read.csv(file.path(root,"prepared/samples.csv"),check.names=FALSE)
axis<-read.csv(file.path(root,"prepared/target_axis_orthology.csv"),check.names=FALSE)
members<-read.csv(file.path(root,"prepared/candidate_membership.csv"),check.names=FALSE)
genes<-readLines(file.path(root,"prepared/candidate_order.txt"))
stopifnot(identical(colnames(y),samples$title),length(genes)==20,nrow(y)==24421,all(y>=0),all(is.finite(y)))
axis$eligible_before_expression_filter<-tolower(axis$eligible_before_expression_filter)=="true"
members$eligible_before_expression_filter<-tolower(members$eligible_before_expression_filter)=="true"
bin<-function(x)findInterval(x,unique(quantile(x,c(.2,.4,.6,.8),type=7)))

fit_model<-function(d,ko) {
 design<-cbind(Intercept=1,KO=as.numeric(ko))
 v<-voom(d,design=design,plot=FALSE)
 fit<-eBayes(lmFit(v,design),robust=TRUE)
 stopifnot(all(is.finite(fit$t[,2])),all(is.finite(fit$s2.post)))
 list(v=v,fit=fit)
}
score_model<-function(model,ko,bg,sets) {
 ix<-match(bg,rownames(model$v$E));stopifnot(!anyNA(ix))
 t<-model$fit$t[ix,2]
 rankscore<-(rank(abs(t),ties.method="average")-.5)/length(t)
 e<-model$v$E[ix,!ko,drop=FALSE]
 mu<-rowMeans(e);vari<-apply(e,1,var);strata<-5*bin(mu)+bin(vari)
 adjusted<-rankscore-ave(rankscore,strata,FUN=mean)
 scores<-vapply(sets,function(s)mean(adjusted[s]),numeric(1))
 unadjusted<-vapply(sets,function(s)mean(rankscore[s]),numeric(1))
 list(scores=scores,unadjusted=unadjusted,gene=data.frame(source_symbol=bg,moderated_t=t,
   control_mean=mu,control_variance=vari,stratum=strata,abs_t_percentile=rankscore,adjusted_score=adjusted))
}

all_summaries<-list();all_perms<-list();all_coverage<-list();all_LOO<-list()
for(variant in c("matched_WT3_KO4","all_WT4_KO4_sensitivity")) {
 keep_sample<-samples$group %in% if(variant=="matched_WT3_KO4") c("WT_matched","KO") else c("WT_matched","WT_extra_flox","KO")
 sam<-samples[keep_sample,];counts<-y[,keep_sample,drop=FALSE];ko<-sam$group=="KO"
 sub<-file.path(out,variant);dir.create(sub,showWarnings=FALSE)
 d<-DGEList(counts)
 keep_gene<-rowSums(cpm(d)>1)>=3
 write.csv(data.frame(source_symbol=rownames(d),keep_CPM_gt1_in_ge3=keep_gene),file.path(sub,"gene_filter.csv"),row.names=FALSE)
 d<-d[keep_gene,,keep.lib.sizes=FALSE];d<-calcNormFactors(d,method="TMM")
 obs<-fit_model(d,ko)
 bg<-axis$source_symbol[axis$eligible_before_expression_filter & axis$source_symbol %in% rownames(d)]
 stopifnot(!anyDuplicated(bg),length(bg)>100)
 sets<-list()
 for(family in c("qualified","raw")) for(g in genes) {
  mm<-members[members$candidate==g & members$family==family,]
  included<-mm$eligible_before_expression_filter & mm$source_symbol %in% bg
  sets[[paste(family,g,sep="__")]]<-match(mm$source_symbol[included],bg)
  stopifnot(sum(included)>0,!anyDuplicated(sets[[paste(family,g,sep="__")]]))
  all_coverage[[length(all_coverage)+1]]<-data.frame(variant=variant,family=family,candidate=g,
   original_n=nrow(mm),source_mapped_n=sum(mm$eligible_before_expression_filter),scored_n=sum(included))
 }
 scored<-score_model(obs,ko,bg,sets)
 write.csv(scored$gene,file.path(sub,"observed_background_scores.csv"),row.names=FALSE)
 f<-obs$fit;se<-f$stdev.unscaled[,2]*sqrt(f$s2.post);critical<-qt(.975,df=f$df.total)
 de<-data.frame(source_symbol=rownames(d),log2FC=f$coefficients[,2],AveExpr=f$Amean,
  moderated_SE=se,df_total=f$df.total,moderated_t=f$t[,2],P=f$p.value[,2],BH=p.adjust(f$p.value[,2],"BH"),
  CI95_low=f$coefficients[,2]-critical*se,CI95_high=f$coefficients[,2]+critical*se)
 write.csv(de,file.path(sub,"all_gene_DE.csv"),row.names=FALSE)
 write.table(obs$v$E,file.path(sub,"voom_logCPM.tsv"),sep="\t",quote=FALSE,col.names=NA)
 write.table(obs$v$weights,file.path(sub,"voom_weights.tsv"),sep="\t",quote=FALSE,col.names=NA)
 write.csv(data.frame(sam,raw_library_counts=colSums(counts),filtered_library_counts=d$samples$lib.size,TMM_factor=d$samples$norm.factors),file.path(sub,"sample_QC.csv"),row.names=FALSE)
 write.csv(cor(obs$v$E),file.path(sub,"sample_correlation.csv"))
 pc<-prcomp(t(obs$v$E),scale.=FALSE)
 write.csv(data.frame(sample=sam$title,PC1=pc$x[,1],PC2=pc$x[,2]),file.path(sub,"sample_PCA.csv"),row.names=FALSE)
 writeLines(c(paste("tested_genes",nrow(d)),paste("background_genes",length(bg)),paste("PCA_PC1_fraction",pc$sdev[1]^2/sum(pc$sdev^2))),file.path(sub,"model_counts.txt"))
 contrasts<-list()
 for(family in c("qualified","raw")) {
  xs<-sets[[paste(family,"XBP1",sep="__")]]
  w<-rep(0,length(bg));w[xs]<-1/length(xs)
  for(g in genes[genes!="XBP1"]) {ii<-sets[[paste(family,g,sep="__")]];w[ii]<-w[ii]-1/(19*length(ii))}
  stopifnot(abs(sum(w))<1e-12)
  write.csv(data.frame(source_symbol=bg,weight=w),file.path(sub,paste0(family,"_global_contrast_weights.csv")),row.names=FALSE)
  contrasts[[family]]<-w
 }
 alloc<-combn(ncol(d),sum(ko));B<-ncol(alloc)
 stopifnot(B==if(variant=="matched_WT3_KO4")35 else 70)
 obs_index<-which(apply(alloc,2,function(z)identical(as.integer(z),as.integer(which(ko)))))
 stopifnot(length(obs_index)==1)
 write.csv(data.frame(allocation=seq_len(B),is_observed=seq_len(B)==obs_index,
   KO_samples=apply(alloc,2,function(z)paste(sam$title[z],collapse=";"))),file.path(sub,"allocations.csv"),row.names=FALSE)
 perm_scores<-matrix(NA_real_,B,length(sets),dimnames=list(NULL,names(sets)))
 for(k in seq_len(B)) {
  pk<-seq_len(ncol(d))%in%alloc[,k]
  model<-if(k==obs_index)obs else fit_model(d,pk)
  sc<-score_model(model,pk,bg,sets)
  perm_scores[k,]<-sc$scores
  for(family in c("qualified","raw")) {
   subset<-sc$scores[paste(family,genes,sep="__")];names(subset)<-genes
   gap<-subset['XBP1']-mean(subset[genes!='XBP1'])
   stopifnot(abs(gap-sum(sc$gene$adjusted_score*contrasts[[family]]))<1e-12)
   all_perms[[length(all_perms)+1]]<-data.frame(variant=variant,allocation=k,is_observed=k==obs_index,family=family,
    XBP1_score=unname(subset['XBP1']),mean_other19=mean(subset[genes!='XBP1']),global_gap=unname(gap),
    own_rank=rank(-subset,ties.method="average")['XBP1'])
  }
  if(k%%10==0) {cat(variant,"permutation",k,"/",B,"\n");flush.console()}
 }
 write.csv(data.frame(allocation=seq_len(B),is_observed=seq_len(B)==obs_index,perm_scores,check.names=FALSE),file.path(sub,"all_candidate_permutation_scores.csv"),row.names=FALSE)
 for(family in c("qualified","raw")) {
  nameset<-paste(family,genes,sep="__");observed<-scored$scores[nameset];names(observed)<-genes
  Tperm<-perm_scores[,paste(family,'XBP1',sep='__')]-rowMeans(perm_scores[,paste(family,genes[genes!='XBP1'],sep='__'),drop=FALSE])
  Tobs<-observed['XBP1']-mean(observed[genes!='XBP1'])
  for(g in genes) {
   ids<-bg[sets[[paste(family,g,sep='__')]]];deix<-match(ids,de$source_symbol)
   all_summaries[[length(all_summaries)+1]]<-data.frame(variant=variant,family=family,candidate=g,scored_n=length(ids),
    adjusted_score=unname(observed[g]),unadjusted_score=scored$unadjusted[paste(family,g,sep='__')],rank=unname(rank(-observed,ties.method='average')[g]),
    DE_BH_lt005_n=sum(de$BH[deix]<.05),median_abs_log2FC=median(abs(de$log2FC[deix])),
    XBP1_minus_mean_other19=unname(Tobs),XBP1_minus_best_other=unname(observed['XBP1']-max(observed[genes!='XBP1'])),
    global_exact_P=mean(Tperm>=Tobs-1e-12),n_allocations=B)
  }
 }
 if(variant=='matched_WT3_KO4') {
  for(omit in seq_len(ncol(d))) {
   dl<-d[,-omit,keep.lib.sizes=FALSE];dl<-calcNormFactors(dl,method='TMM')
   fitl<-fit_model(dl,ko[-omit]);scl<-score_model(fitl,ko[-omit],bg,sets)
   for(family in c('qualified','raw')) {
    ss<-scl$scores[paste(family,genes,sep='__')];names(ss)<-genes
    all_LOO[[length(all_LOO)+1]]<-data.frame(omitted_sample=sam$title[omit],omitted_group=sam$group[omit],family=family,
       XBP1_score=unname(ss['XBP1']),mean_other19=mean(ss[genes!='XBP1']),global_gap=unname(ss['XBP1']-mean(ss[genes!='XBP1'])),own_rank=unname(rank(-ss)['XBP1']))
   }
  }
 }
 cat(variant,"COMPLETE; tested",nrow(d),"background",length(bg),"\n");flush.console()
}
write.csv(do.call(rbind,all_summaries),file.path(out,"all20_candidate_summary.csv"),row.names=FALSE)
write.csv(do.call(rbind,all_perms),file.path(out,"global_contrast_all_permutations.csv"),row.names=FALSE)
write.csv(do.call(rbind,all_coverage),file.path(out,"candidate_coverage.csv"),row.names=FALSE)
write.csv(do.call(rbind,all_LOO),file.path(out,"leave_one_pool_out.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_sessionInfo.txt"))
writeLines(format(Sys.time(),tz='UTC',usetz=TRUE),file.path(out,"COMPLETE.txt"))
print(do.call(rbind,all_summaries)[vapply(all_summaries,function(x)x$candidate=='XBP1',logical(1)),])
