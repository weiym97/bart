rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)

extract_posterior <- function(subjs,result){
  n_subj <- length(subjs)
  param_name <- unique(gsub("\\[.*","",x=row.names(result)))
  posterior <-data.frame(matrix(ncol=length(param_name)+1,nrow=n_subj, dimnames=list(NULL, c('subjID',param_name))))
  for (i in 1:n_subj) {
    posterior[i,'subjID'] <- subjs[i]
    for (j in 1:length(param_name)){
      temp_name <- paste(param_name[j],'[',i,']',sep='')
      posterior[i,param_name[j]] = result[temp_name,'mean']
    }
  }
  return(posterior)
}

data_type <- args[1]
model_name <- args[2]
data_file_name <- args[3]

library("hBayesDM")

#cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling 4 simulations in Stan... \n")
output = bart_par4(data=paste('data/',data_type,'/',data_file_name,'.txt',sep=''), 
                     niter=2000, nwarmup=1000, nchain=4, ncore=4)

#cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)
cat("It took",as.character.Date(endTime - startTime), "\n")

# save the result
fit <- output$fit
save(fit,file=paste('fit_result_Ji/',model_name,'_',data_file_name,'.Rdata',sep=''))

param_name <- c('phi','eta','gam','tau')
result_summary<-as.data.frame(rstan::summary(fit,pars=param_name)$summary)
posterior_mean <- extract_posterior(subjs,result_summary)
write.table(posterior_mean,paste('fit_result_Ji/summary_',model_name,'_',data_file_name,'.txt',sep=''),quote=F,row.names=F)
