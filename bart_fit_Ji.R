rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)

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
