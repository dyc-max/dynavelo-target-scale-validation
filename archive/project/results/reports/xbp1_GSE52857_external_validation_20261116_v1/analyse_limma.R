args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1],winslash="/")
.libPaths(c(file.path(root,"R_library"),.libPaths()))
library(limma)
library(jsonlite)
out <- file.path(root,"outputs")
if (file.exists(file.path(out,"candidate_support_all20.csv"))) stop("Refuse overwrite")
dir.create(out,showWarnings=FALSE)
y <- as.matrix(read.delim(file.path(root,"prepared","mouse_gene_expression.tsv"),row.names=1,check.names=FALSE))
storage.mode(y) <- "double"
stopifnot(all(is.finite(y)),ncol(y)==12,all(y>0),max(y)<30)
samples <- read.csv(file.path(root,"prepared","samples.csv"),stringsAsFactors=FALSE,fileEncoding="UTF-8-BOM")
stopifnot(identical(colnames(y),samples$sample_id))
bgids <- readLines(file.path(root,"prepared","background_mouse_ids.txt"))
bg <- match(bgids,rownames(y));stopifnot(!anyNA(bg),!anyDuplicated(bg))
members <- read.csv(file.path(root,"prepared","candidate_membership.csv"),stringsAsFactors=FALSE,fileEncoding="UTF-8-BOM")
members$included_in_score <- tolower(as.character(members$included_in_score))=="true"
genes <- readLines(file.path(root,"prepared","candidate_order.txt"))
families <- c("qualified","raw")
sets <- list()
for (f in families) for(g in genes) {
  ids <- unique(members$mouse_ensembl[members$candidate==g & members$family==f & members$included_in_score])
  sets[[paste(f,g,sep="__")]] <- match(ids,bgids)
}
stopifnot(all(vapply(sets,function(x)length(x)>0 && !anyNA(x),logical(1))))

fit_model <- function(v,ko) {
  design <- cbind(Intercept=1,KO=as.numeric(ko))
  fit <- eBayes(lmFit(v,design),trend=TRUE,robust=TRUE)
  list(fit=fit,t=fit$t[,2],p=fit$p.value[,2],logFC=fit$coefficients[,2])
}
score_model <- function(v,ko) {
  fit <- fit_model(v,ko)
  t <- fit$t[bg]
  rankscore <- (rank(abs(t),ties.method="average")-.5)/length(t)
  controls <- v[bg,!ko,drop=FALSE]
  mu <- rowMeans(controls);variance <- apply(controls,1,var)
  bin <- function(x) findInterval(x,unique(quantile(x,c(.2,.4,.6,.8),type=7)))
  strata <- 5*bin(mu)+bin(variance)
  adjusted <- rankscore-ave(rankscore,strata,FUN=mean)
  scores <- vapply(sets,function(index)mean(adjusted[index]),numeric(1))
  unadjusted <- vapply(sets,function(index)mean(rankscore[index]),numeric(1))
  list(fit=fit,scores=scores,unadjusted=unadjusted,strata=strata,rankscore=rankscore,adjusted=adjusted)
}
all_results <- list();gene_results <- list();all_permutations <- list();gaps <- list();specificity <- list();unique_rows <- list();sample_qc <- list();target_rows <- list()
for (subtype in c("CD8alpha_DC","CD11b_DC")) {
  ix <- which(samples$subtype==subtype)
  v <- y[,ix,drop=FALSE];ko <- samples$genotype[ix]=="KO"
  stopifnot(length(ix)==6,sum(ko)==3)
  observed <- score_model(v,ko)
  alternatives <- combn(6,3)
  permutations <- matrix(NA_real_,20,length(sets),dimnames=list(NULL,names(sets)))
  for(k in seq_len(ncol(alternatives))) {
    permko <- seq_len(6)%in%alternatives[,k]
    z <- score_model(v,permko)
    permutations[k,] <- z$scores
  }
  # For this unsigned statistic, complementary label allocations need not have
  # identical adjusted scores because control-only strata are recalculated.
  # Exact20 allocations still bound minimum nonzero P at1/20.
  identical_labels <- which(apply(alternatives,2,function(x)identical(as.integer(x),as.integer(which(ko)))))
  stopifnot(length(identical_labels)==1,max(abs(permutations[identical_labels,]-observed$scores))<1e-10)
  de <- data.frame(mouse_ensembl=rownames(y),subtype=subtype,log2FC_KO_minus_WT=observed$fit$logFC,
                   moderated_t=observed$fit$t,p_raw=observed$fit$p,FDR_BH_all_measured_genes=p.adjust(observed$fit$p,"BH"),
                   mean_log2_WT=rowMeans(v[,!ko,drop=FALSE]),mean_log2_KO=rowMeans(v[,ko,drop=FALSE]))
  gene_results[[subtype]] <- de
  for(j in seq_len(6)) sample_qc[[length(sample_qc)+1]] <- data.frame(sample_id=colnames(v)[j],subtype=subtype,
          genotype=ifelse(ko[j],"KO","WT"),median_log2_expression=median(v[,j]),q05=quantile(v[,j],.05),q95=quantile(v[,j],.95))
  for(f in families) {
    keys <- paste(f,genes,sep="__");values <- observed$scores[keys];unadj <- observed$unadjusted[keys]
    ranks <- rank(-values,ties.method="average")
    for(i in seq_along(genes)) {
      key <- keys[i];index <- sets[[key]];other <- values[-i]
      p <- mean(permutations[,key]>=values[i]-1e-12)
      ownDE <- de[bg[index],]
      all_results[[length(all_results)+1]] <- data.frame(subtype=subtype,family=f,candidate=genes[i],mapped_scored_targets_n=length(index),
           control_stratified_abs_t_rank_score=unname(values[i]),support_rank_among20=unname(ranks[i]),
           unadjusted_abs_t_percentile_mean=unname(unadj[i]),unadjusted_rank_among20=unname(rank(-unadj,ties.method="average")[i]),
           median_abs_log2FC=median(abs(ownDE$log2FC_KO_minus_WT)),DE_FDR005_targets_n=sum(ownDE$FDR_BH_all_measured_genes<.05),
           exact20_label_permutation_p=p,own_minus_best_other=unname(values[i]-max(other)),
           own_minus_median_other=unname(values[i]-median(other)),
           best_wrong_candidate=paste(genes[-i][other==max(other)],collapse=";"),
           unique_best=all(values[i]>other),status="POST_HOC_EXTERNAL_PERTURBATION_RESPONSE_NOT_DIRECT_EDGE_PROOF")
    }
    xkey <- paste(f,"XBP1",sep="__");wrong <- setdiff(keys,xkey)
    own_gap <- observed$scores[xkey]-max(observed$scores[wrong])
    own_median_gap <- observed$scores[xkey]-median(observed$scores[wrong])
    perm_gap <- permutations[,xkey]-apply(permutations[,wrong,drop=FALSE],1,max)
    perm_median_gap <- permutations[,xkey]-apply(permutations[,wrong,drop=FALSE],1,median)
    gaps[[length(gaps)+1]] <- data.frame(subtype=subtype,family=f,XBP1_minus_best_wrong=unname(own_gap),
       exact20_gap_vs_best_p=mean(perm_gap>=own_gap-1e-12),XBP1_minus_median_wrong=unname(own_median_gap),
       exact20_gap_vs_median_p=mean(perm_median_gap>=own_median_gap-1e-12))
    xs <- sets[[xkey]];otherunion <- unique(unlist(sets[wrong]));shared <- intersect(xs,otherunion);exclusive <- setdiff(xs,otherunion)
    for(label in c("all_XBP1","shared_with_any_other19","exclusive_to_XBP1")) {
      idx <- switch(label,all_XBP1=xs,shared_with_any_other19=shared,exclusive_to_XBP1=exclusive)
      unique_rows[[length(unique_rows)+1]] <- data.frame(subtype=subtype,family=f,part=label,n=length(idx),
             stratified_score=if(length(idx))mean(observed$adjusted[idx]) else NA,
             mouse_ensembl_ids=paste(bgids[idx],collapse=";"))
    }
    for(j in which(genes!="XBP1")) {
      os <- sets[[keys[j]]];xonly <- setdiff(xs,os);oonly <- setdiff(os,xs)
      specificity[[length(specificity)+1]] <- data.frame(subtype=subtype,family=f,wrong_candidate=genes[j],
         shared_targets_n=length(intersect(xs,os)),XBP1_only_n=length(xonly),wrong_only_n=length(oonly),
         XBP1_only_score=if(length(xonly))mean(observed$adjusted[xonly]) else NA,
         wrong_only_score=if(length(oonly))mean(observed$adjusted[oonly]) else NA,
         exclusive_score_difference=if(length(xonly)&&length(oonly))mean(observed$adjusted[xonly])-mean(observed$adjusted[oonly]) else NA)
    }
  }
  for(k in seq_len(20)) for(key in names(sets)) all_permutations[[length(all_permutations)+1]] <- data.frame(subtype=subtype,
       allocation=k,KO_sample_ids=paste(colnames(v)[alternatives[,k]],collapse=";"),is_observed=k==identical_labels,
       candidate_set=key,stratified_abs_t_rank_score=permutations[k,key])
  for(k in seq_len(nrow(members))) {
    row <- members[k,];j <- match(row$mouse_ensembl,rownames(y));bidx <- match(row$mouse_ensembl,bgids)
    evidence <- if(!is.na(j))de[j,c("log2FC_KO_minus_WT","moderated_t","p_raw","FDR_BH_all_measured_genes","mean_log2_WT","mean_log2_KO")] else
      data.frame(log2FC_KO_minus_WT=NA,moderated_t=NA,p_raw=NA,FDR_BH_all_measured_genes=NA,mean_log2_WT=NA,mean_log2_KO=NA)
    target_rows[[length(target_rows)+1]] <- cbind(row,subtype=subtype,evidence,
       control_mean_variance_stratum=if(!is.na(bidx))observed$strata[bidx] else NA,
       abs_t_percentile=if(!is.na(bidx))observed$rankscore[bidx] else NA,
       stratified_rank_residual=if(!is.na(bidx))observed$adjusted[bidx] else NA)
  }
  cat(subtype,"completed: ",sum(de$FDR_BH_all_measured_genes<.05),"gene-level BH<0.05\n")
}
results <- do.call(rbind,all_results)
results$set_p_BH_40_within_family <- ave(results$exact20_label_permutation_p,results$family,FUN=function(x)p.adjust(x,"BH"))
gaps <- do.call(rbind,gaps)
gaps$gap_best_Holm_two_subtypes_within_family <- ave(gaps$exact20_gap_vs_best_p,gaps$family,FUN=function(x)p.adjust(x,"holm"))
write.csv(results,file.path(out,"candidate_support_all20.csv"),row.names=FALSE)
write.csv(do.call(rbind,gene_results),file.path(out,"all_measured_mouse_genes_DE.csv"),row.names=FALSE)
write.csv(do.call(rbind,all_permutations),file.path(out,"all_exact_label_permutations.csv"),row.names=FALSE)
write.csv(gaps,file.path(out,"XBP1_specificity_gap_tests.csv"),row.names=FALSE)
write.csv(do.call(rbind,specificity),file.path(out,"XBP1_vs_each_other_exclusive_targets.csv"),row.names=FALSE)
write.csv(do.call(rbind,unique_rows),file.path(out,"XBP1_shared_and_unique_targets.csv"),row.names=FALSE)
write.csv(do.call(rbind,target_rows),file.path(out,"all_frozen_target_perturbation_evidence.csv"),row.names=FALSE)
write.csv(do.call(rbind,sample_qc),file.path(out,"sample_QC.csv"),row.names=FALSE)
write.csv(results[results$candidate=="XBP1",],file.path(out,"XBP1_summary.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_sessionInfo.txt"))
cat(toJSON(results[results$candidate=="XBP1",],pretty=TRUE,digits=12),"\n")
