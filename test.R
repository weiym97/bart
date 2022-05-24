# Set up the true parameter values
a <- c(0.8,1)
b <- c(0.2,0.1)
sigma <- 0.2

#Simulate data
x <- (1:1000)/100
N <-length(x)
ypred <- a[1] *exp(-b[1] * x) + a[2] * exp(-b[2] * x)
y <- ypred * exp(rnorm(N,0,sigma))
