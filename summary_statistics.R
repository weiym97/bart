rm(list=ls())

return_param <- function(model_name){
  params=switch(EXPR=model_name,
                FourparamBart=c('phi','eta','gamma','tau'),
                EWBart_1 = c('psi','xi','rho','lambda','tau'),
                EWMVBart_1 = c('psi','xi','rho','lambda','tau'),
                PTBart_9=c('psi','xi','gamma','tau','lambda'),
                PTBart_10=c('psi','xi','gamma','tau','lambda'),
                PTBart_20=c('psi','xi','lambda','tau'),
                PTBart_100=c('psi','xi','gamma','tau','lambda'),
                PTBart_101=c('psi','xi','gamma','tau','lambda','alpha'),
                PTBart_102=c('psi','xi','gamma','tau','lambda','alpha','beta'),
                PTBart_103=c('psi','xi','gamma','tau','lambda','alpha'),
                PTBart_104=c('psi','xi','gamma','tau','lambda','alpha'),
                PTBart_105=c('psi','xi','gamma','tau','lambda','alpha','beta'),
                PTBart_106=c('psi','xi','gamma','tau','lambda','alpha','beta'),
                STLBart = c('omega_0','vwin','vloss','tau'),
                STLDBart = c('omega_0','vwin','vloss','alpha','tau'),
                )
  return(params)
}
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

df <- read.table(paste('data/',data_type, '/',data_file_name,'.txt',sep=''),header=T)
subjs <- unique(df$subjID)
load(paste('fit_result/',model_name,'_',data_file_name,'.Rdata',sep=''))

param_name <- c('psi','xi','gamma','tau','lambda')
result_summary<-as.data.frame(rstan::summary(fit,pars=param_name)$summary)
posterior_mean <- extract_posterior(subjs,result_summary)
write.table(posterior_mean,paste('fit_result/summary_',model_name,'_',data_file_name,'.txt',sep=''),quote=F,row.names=F)