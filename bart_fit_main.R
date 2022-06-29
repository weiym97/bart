###s######################
### construct data
###########################
rm(list=ls())

preprocessing <- function(data){
  # Only preserve trials with pump number no less than 2
  data <- data[data$pumps>=2,]
  return(data)
}

df <- read.table('data/MDD_13.txt',header=T)
df <- preprocessing(df)


subjs <- unique(df$subjID)
n_subj <- length(subjs)
t_max <- max(df$trial)
t_subjs <- array(0, n_subj)
pumps     <- array(0, c(n_subj, t_max))
explosion <- array(0, c(n_subj, t_max))
L <- array(0, c(n_subj, t_max))

# Write from df to the data arrays
for (i in 1:n_subj) {
  subj <- subjs[i]
  DT_subj <- subset(df, subjID == subj)
  t_subjs[i] <- length(DT_subj$trial)
  t <- t_subjs[i]
  pumps[i, 1:t]     <- DT_subj$pumps
  explosion[i, 1:t] <- DT_subj$explosion
  L[i, 1:t] <- pumps[i, 1:t] + 1 - explosion[i, 1:t]
}

#L[L==12] <- 11

# r=0:max(pumps)
r_accu = c(0.00, 0.00,0.05, 0.15, 0.25, 0.55, 0.95, 1.45, 2.05, 2.75, 3.45, 4.25, 5.15, 6.00)
r      = c()
for (j in 1:length(r_accu)-1) {
  r[j] <- r_accu[j+1]-r_accu[j]
}

# Wrap into a list for Stan
dataList <- list(
  N         = n_subj,
  T         = t_max,
  Tsubj     = t_subjs,
  P         = length(r),
  pumps     = pumps,
  explosion = explosion,
  r         = r,
  r_accu    = r_accu,
  L         = L
)

###############################
### fit model
###############################
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
options(mc.cores = 4)

#nIter     = 2000   # 2000 for test, 4000 for real data
nIter     =4000 
nChains   = 4 
nWarmup   = floor(nIter/2)
nThin     = 1

modelFile = './PTBart_3.stan'
cat("Estimating", modelFile, "model... \n")
startTime = Sys.time(); print(startTime)
cat("Calling", nChains, "simulations in Stan... \n")
fit = stan(
  modelFile,
  data    = dataList, 
  chains  = nChains,
  iter    = nIter,
  warmup  = nWarmup,
  thin    = nThin,
  #init    = "random",
  # control = list(adapt_delta = 0.999, max_treedepth = 20),
  # control  = list(adapt_delta=0.999, max_treedepth=100),
  #seed    = 233
)
cat("Finishing", modelFile, "model simulation ... \n")
endTime = Sys.time(); print(endTime)  
cat("It took",as.character.Date(endTime - startTime), "\n")

# save the result
save(fit,file="fit_result/PTBart_3_second_try.Rdata")

