args<-commandArgs(trailingOnly=TRUE);root<-normalizePath(args[1],winslash='/');out<-file.path(root,'analysis')
if(dir.exists(out))stop('Refuse overwrite')
dir.create(out)
.libPaths(c('D:/dyc_7/DynaVelo_Research/results/reports/xbp1_GSE52857_external_validation_20261116_v1/R_library',.libPaths()))
suppressPackageStartupMessages(library(limma));stopifnot(as.character(packageVersion('limma'))=='3.68.5')
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
csv<-function(x,p)write.csv(x,p,row.names=FALSE,na='')
matout<-function(x,p)write.table(cbind(symbol=rownames(x),as.data.frame(x,check.names=FALSE)),p,sep='\t',quote=FALSE,row.names=FALSE,na='')
cfg<-data.frame(dataset=c('GSE58743','GSE84388','GSE146288','GSE146288'),contrast=c('K562_CEBPA_induced','PreGM_Cebpa_KO','HSPC_Cebpa_KO_untreated','HSPC_Cebpa_KO_FLT3L4h'),time=c('48h','18d_after_pIC','untreated','FLT3L4h'),activation_sign=c(1,-1,-1,-1),stringsAsFactors=FALSE)
all<-list();allsamples<-list();probes<-list();allqc<-list()
for(k in seq_len(nrow(cfg))) {
 c<-cfg[k,];cat('START',c$contrast,'\n');flush.console();dest<-file.path(out,c$contrast);dir.create(dest)
 sam<-read.csv(file.path(root,'prepared',paste0(c$dataset,'_samples.csv')),check.names=FALSE);sam<-sam[sam$time==c$time,];y<-as.matrix(read.delim(file.path(root,'prepared',paste0(c$dataset,'_gene_expression.tsv')),row.names=1,check.names=FALSE));y<-y[,sam$column,drop=FALSE];storage.mode(y)<-'double';mut<-sam$mutant==1
 stopifnot(ncol(y)==if(c$dataset=='GSE84388')6 else 8,all(is.finite(y)),sum(mut)==ncol(y)/2)
 design<-cbind(Intercept=1,Altered=as.numeric(mut));fit<-eBayes(lmFit(y,design),robust=TRUE,trend=TRUE)
 se<-fit$stdev.unscaled[,2]*sqrt(fit$s2.post);crit<-qt(.975,fit$df.total)
 de<-data.frame(symbol=rownames(y),log2_effect=fit$coefficients[,2],SE=se,CI95_low=fit$coefficients[,2]-crit*se,CI95_high=fit$coefficients[,2]+crit*se,t=fit$t[,2],P=fit$p.value[,2],BH_all_genes=p.adjust(fit$p.value[,2],'BH'),Amean=fit$Amean,s2_post=fit$s2.post,df_total=fit$df.total,stdev_unscaled=fit$stdev.unscaled[,2],residual_variance=fit$sigma^2)
 stopifnot(all(is.finite(de$t)),all(is.finite(de$P)));csv(de,file.path(dest,'all_gene_effects.csv'));matout(y,file.path(dest,'normalized_expression.tsv'));csv(sam,file.path(dest,'samples.csv'))
 target<-if(c$dataset=='GSE58743')c('CEBPA','MPO','CSF3R') else c('Cebpa','Mpo','Csf3r');rr<-de[match(target,de$symbol),];rr$human_gene<-c('CEBPA','MPO','CSF3R');rr$dataset<-c$dataset;rr$contrast<-c$contrast;rr$perturbation<-if(c$activation_sign==1)'gain' else 'loss';rr$n_control<-sum(!mut);rr$n_altered<-sum(mut);rr$analysis_status<-'INDIVIDUAL_SAMPLE_LIMMA';rr$activation_oriented_effect<-c$activation_sign*rr$log2_effect;rr$activation_oriented_CI_low<-if(c$activation_sign==1)rr$CI95_low else -rr$CI95_high;rr$activation_oriented_CI_high<-if(c$activation_sign==1)rr$CI95_high else -rr$CI95_low;rr$mean_control_log2<-rowMeans(y[target,!mut,drop=FALSE]);rr$mean_altered_log2<-rowMeans(y[target,mut,drop=FALSE]);rr$control_signal_percentile<-(rank(rowMeans(y[,!mut,drop=FALSE]))[match(target,rownames(y))]-.5)/nrow(y);rr$exported_mean_control<-NA_real_;rr$exported_mean_altered<-NA_real_;all[[k]]<-rr
 targexpr<-data.frame(symbol=rep(target,each=ncol(y)),human_gene=rep(c('CEBPA','MPO','CSF3R'),each=ncol(y)),GSM=rep(colnames(y),times=3),group=rep(sam$condition,times=3),value_log2=as.vector(t(y[target,,drop=FALSE])));targexpr$contrast<-c$contrast;targexpr$dataset<-c$dataset;allsamples[[k]]<-targexpr
 cor<-cor(y);csv(data.frame(sample=colnames(y),as.data.frame(cor,check.names=FALSE)),file.path(dest,'sample_correlations.csv'));rv<-apply(y,1,var);pc<-prcomp(t(y[order(rv,decreasing=TRUE)[seq_len(min(2000,length(rv)))],,drop=FALSE]),scale.=FALSE)
 allqc[[k]]<-data.frame(contrast=c$contrast,dataset=c$dataset,GSM=colnames(y),group=sam$condition,median_log2=apply(y,2,median),PC1=pc$x[,1],PC2=pc$x[,2],mapped_genes=nrow(y));cat('DONE',c$contrast,'genes',nrow(y),'\n')
}
inference<-do.call(rbind,all);cols<-names(inference)
thp<-read.delim(file.path(root,'prepared/GSE119551_exported_group_means.tsv'),check.names=FALSE)
for(ctx in c('E','V')) {
 targets<-c('CEBPA','MPO','CSF3R');stopifnot(all(vapply(targets,function(g)sum(thp$Gene==g)==1,logical(1))))
 for(g in targets) {
  v<-thp[thp$Gene==g,];mc<-v[[paste0('avg_NC1_',ctx)]];ma<-v[[paste0('avg_CEBPA_',ctx)]];effect<-if(mc>0 && ma>0)log2(ma/mc) else NA_real_
  z<-as.data.frame(setNames(lapply(cols,function(x)NA),cols));z$symbol<-g;z$human_gene<-g;z$dataset<-'GSE119551';z$contrast<-paste0('THP1_CEBPA_KD_',if(ctx=='E')'solvent' else 'vitD');z$perturbation<-'loss';z$n_control<-3;z$n_altered<-3;z$analysis_status<-'EXPORTED_GROUP_MEANS_ONLY_UNITS_UNRESOLVED';z$log2_effect<-effect;z$activation_oriented_effect<--effect;z$exported_mean_control<-mc;z$exported_mean_altered<-ma;inference<-rbind(inference,z)
 }
}
inference$is_requested_edge<-inference$human_gene %in% c('MPO','CSF3R');inference$P_for_Holm12<-NA_real_;inference$Holm12_P<-NA_real_;ix<-which(inference$is_requested_edge);stopifnot(length(ix)==12,nrow(inference)==18)
inference$P_for_Holm12[ix]<-ifelse(is.na(inference$P[ix]),1,inference$P[ix]);inference$Holm12_P[ix]<-p.adjust(inference$P_for_Holm12[ix],'holm')
inference$expected_positive_direction<-ifelse(is.na(inference$activation_oriented_effect),NA,inference$activation_oriented_effect>0)
csv(inference,file.path(out,'ALL_18_GENE_READOUTS.csv'));csv(inference[inference$is_requested_edge,],file.path(out,'REQUESTED_12_EDGE_ENDPOINTS.csv'));csv(do.call(rbind,allsamples),file.path(out,'ALL_INDIVIDUAL_TARGET_EXPRESSION.csv'));csv(do.call(rbind,allqc),file.path(out,'SAMPLE_QC.csv'))
writeLines(format(Sys.time(),tz='UTC',usetz=TRUE),file.path(out,'COMPLETE.txt'));cat('ALL_COMPLETED\n')
