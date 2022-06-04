
data {
  int<lower=1> N;            // Number of subjects
  int<lower=1> T;            // Maximum number of trials
  int<lower=0> Tsubj[N];     // Number of trials for each subject
  int<lower=2> P;            // Number of max pump
  int<lower=0> L[N, T];      // Number of decision times
  int<lower=0> pumps[N, T];  // Number of pump
  int<lower=0,upper=1> explosion[N, T];  // Whether the balloon exploded (0 or 1)
  real r[P];
  real r_accu[P+1];
}

transformed data{
  // Whether a subject pump the button or not (0 or 1)
  int d[N, T, P];

  for (j in 1:N) {
    for (k in 1:Tsubj[j]) {
      for (l in 1:P) {
        if (l <= pumps[j, k])
          d[j, k, l] = 1;
        else
          d[j, k, l] = 0;
      }
    }
  }
}

parameters {
  // Group-level parameters
  vector[6] mu_pr;
  vector<lower=0>[6] sigma;

  // Normally distributed error for Matt trick
  vector[N] phi_pr;
  vector[N] eta_pr;
  vector[N] gam_pr;
  vector[N] lambda_pr;
  vector[N] theta_pr;
  vector[N] tau_pr;
}

transformed parameters {
  // Subject-level parameters with Matt trick
  vector<lower=0,upper=1>[N] phi;
  vector<lower=0>[N] eta;
  vector<lower=0>[N] gam;
  vector<lower=0>[N] lambda;
  vector<lower=0,upper=1>[N] theta;
  vector<lower=0>[N] tau;

  phi = Phi_approx(mu_pr[1] + sigma[1] * phi_pr);
  eta = exp(mu_pr[2] + sigma[2] * eta_pr);
  gam = exp(mu_pr[3] + sigma[3] * gam_pr);
  lambda = exp(mu_pr[4] + sigma[4] * lambda_pr);
  theta = Phi_approx(mu_pr[5] + sigma[5] * theta_pr);
  tau = exp(mu_pr[6] + sigma[6] * tau_pr);
}

model {
  // Prior
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 5);

  phi_pr ~ normal(0, 1);
  eta_pr ~ normal(0, 1);
  gam_pr ~ normal(0, 1);
  lambda_pr ~ normal(0, 1);
  theta_pr ~ normal(0, 1);
  tau_pr ~ normal(0, 1);

  // Likelihood
  for (j in 1:N) {
    // Initialize n_succ and n_pump for a subject
    int n_succ = 0;  // Number of successful pumps
    int n_pump = 0;  // Number of total pumps

    for (k in 1:Tsubj[j]) {
      real p_burst;  // Belief on a balloon to be burst
      real omega;    // Optimal number of pumps
      real Loss_aver = 0;

      p_burst = 1 - ((phi[j] + eta[j] * n_succ) / (1 + eta[j] * n_pump));
      omega = -gam[j] / log1m(p_burst);
      
      

      // Calculate likelihood with bernoulli distribution
      for (l in 1:L[j,k]){
        d[j, k, l] ~ bernoulli_logit(tau[j] * (omega - lambda[j] * Loss_aver- l));
      }
      if (explosion[j,k] ==0) 
          Loss_aver = Loss_aver + theta[j] * (r_accu[pumps[j,k] + 1] - Loss_aver);
      else
          Loss_aver = (1 - theta[j]) * Loss_aver;
      
      // Update n_succ and n_pump after each trial ends
      n_succ += pumps[j, k] - explosion[j, k];
      n_pump += pumps[j, k];
    }
  }
}

generated quantities {
  // Actual group-level mean
  real<lower=0, upper=1> mu_phi = Phi_approx(mu_pr[1]);
  real<lower=0> mu_eta = exp(mu_pr[2]);
  real<lower=0> mu_gam = exp(mu_pr[3]);
  real<lower=0> mu_lambda = exp(mu_pr[4]);
  real<lower=0,upper=1> mu_theta = Phi_approx(mu_pr[5]);
  real<lower=0> mu_tau = exp(mu_pr[6]);

  // Log-likelihood for model fit
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T, P];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (j in 1:N)
    for (k in 1:T)
      for(l in 1:P)
        y_pred[j, k, l] = -1;

  { // Local section to save time and space
    for (j in 1:N) {
      int n_succ = 0;
      int n_pump = 0;
      real Loss_aver = 0;

      log_lik[j] = 0;

      for (k in 1:Tsubj[j]) {
        real p_burst;  // Belief on a balloon to be burst
        real omega;    // Optimal number of pumps

        p_burst = 1 - ((phi[j] + eta[j] * n_succ) / (1 + eta[j] * n_pump));
        omega = -gam[j] / log1m(p_burst);

        

        for (l in 1:L[j,k]) {
          log_lik[j] += bernoulli_logit_lpmf(d[j, k, l] | tau[j] * (omega - lambda[j] * Loss_aver- l));
          y_pred[j, k, l] = bernoulli_logit_rng(tau[j] * (omega -  lambda[j] * Loss_aver - l));
        }

        n_succ += pumps[j, k] - explosion[j, k];
        n_pump += pumps[j, k];
        if (explosion[j,k] ==0)
          Loss_aver = Loss_aver + theta[j] * (r_accu[pumps[j,k] + 1] - Loss_aver);
        else
          Loss_aver = (1 - theta[j]) * Loss_aver;
      }
    }
  }
}

