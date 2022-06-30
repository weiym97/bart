rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)

param_name <- c("psi","xi","gam","tau",'lambda')
#file_name <- 'BASEBart_MDD_13'
file_name <- args[1]

load(paste('fit_result/',file_name,'.Rdata',sep=''))

fit_result <- rstan::summary(fit,pars=param_name)$summary
fit_result_dataframe<-as.data.frame(fit_result)
write.table(fit_result_dataframe,paste('fit_result/',file_name,'.txt',seq=''))
