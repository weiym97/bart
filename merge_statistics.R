rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)

model_name <- args[1]
data_name <- args[2]
n_file <- as.integer(args[3])

result <- read.table(paste('fit_result/summary_',model_name,'_',data_name,'_1.txt',sep=''),header=T)
for (i in 2:n_file){
  new_result <-read.table(paste('fit_result/summary_',model_name,'_',data_name,'_',i,'.txt',sep=''),header=T)
  result <- rbind(result,new_result)
}
write.table(result,paste('fit_result/summary_',model_name,'_',data_name,'.txt',sep=''),quote=F,row.names=F)