rm(list=ls())


preprocessing <- function(data){
  # Only preserve trials with pump number no less than 2
  data <- data[data$pumps>=2,]
  return(data)
}

compute_BART_score <- function(data){
  data <- data[data$explosion==0,]
  subjID <- unique(data$subjID)
  BART_score <- rep(x=0,times=length(subjID))
  for (i in 1:length(subjID)){
    data_subj <- data[data$subjID == subjID[i],]
    BART_score[i] <- mean(data_subj$pumps)
    
  }
  return(BART_score)
}

compute_withdraw_prop <- function(data){
  subjID<-unique(data$subjID)
  withdraw_prop = rep(x=0,times=length(subjID))
  for (i in 1:length(subjID)){
    data_subj <- data[data$subjID==subjID[i],]
    count_num <- 0
    total_num <- 0
    for (j in 2:nrow(data_subj)){
      if ((data_subj$explosion[j] == 0)){ #&& (data_subj$explosion[j-1] == 0)){
        total_num <- total_num + 1
        if (data_subj$pumps[j] < data_subj$pumps[j-1]){
          count_num <- count_num +1
        }
      }
      
    }
    if (total_num>0){
      withdraw_prop[i] <- count_num/total_num
    }
    else{
      withdraw_prop[i] <- 0
    }
    
    
  }
  return(withdraw_prop)
}
data_name='MDD_13'

data <- read.table(file=paste('data/',data_name,'.txt',sep=''),header=T)

data <- preprocessing(data)
#write.csv(data,file='data/MDD_13_preprocessing.csv',quote=F)
BART_score <- compute_BART_score(data)
withdraw_prop<- compute_withdraw_prop(data)
length(BART_score)
length(withdraw_prop)

BART_statistics=data.frame(subjID=unique(data$subjID),BART_score=BART_score,withdraw_prop=withdraw_prop)
write.csv(BART_statistics,paste(data_name,'_statistics.csv',sep=''),row.names=F)
#cor(BART_statistics$BART_score,BART_statistics$withdraw_prop)