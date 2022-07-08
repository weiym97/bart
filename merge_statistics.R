rm(list=ls())
args <- commandArgs(trailingOnly = TRUE)

model_name <- args[1]
data_name <- args[2]
n_file <- as.integer(args[3])

result <- read.table(paste('fit_result/summary',model_name,'_',data_name,'_1.txt',sep=''))
for (i in 2:n_file){
  new_result <-read.table(paste('fit_result/',model_name,'_',data_name,'_',i,'.txt',sep=''))
  result <- rbind(result,new_result)
}
write.table(result,paste('fit_result/summary',model_name,'_',data_name,'.txt',sep=''))