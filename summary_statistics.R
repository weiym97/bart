rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)

extract_posterior <- function(subjs,param_name,result){
  n_subj <- length(subjs)
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

param_name <- c("psi","xi","gam","tau",'lambda')
#file_name <- 'BASEBart_MDD_13'
file_name <- args[1]

load(paste('fit_result/',file_name,'.Rdata',sep=''))

fit_result <- rstan::summary(fit,pars=param_name)$summary
fit_result_dataframe<-as.data.frame(fit_result)
write.table(fit_result_dataframe,paste('fit_result/',file_name,'.txt',seq=''))

result_summary <-read.table('fit_result/PTBart_10_MDD_13.txt',header=T)
df <- read.table('data/MDD_13.txt',header=T)
param_name <-c("psi","xi","gam","tau",'lambda')
subjs <- unique(df$subjID)
posterior_mean <- extract_posterior(subjs,param_name,result_summary)
write.table(posterior_mean,'fit_result/fit_result_test.txt',quote=F,row.names=F)