param_name <- c("omega_0","alpha","lambda","tau")
file_name <- 'BASEBart_MDD_13'

load(paste('fit_result/',file_name,'.Rdata',sep=''))

fit_result <- rstan::summary(fit,pars=param_name)$summary
fit_result_dataframe<-as.data.frame(fit_result)
write.csv(fit_result_dataframe,paste('fit_result/',file_name,'.csv',seq=''))
