
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
  vector[4] mu_pr;
  vector<lower=0>[4] sigma;

  // Normally distributed error for Matt trick
  vector[N] omega_0_pr;
  vector[N] vwin_pr;
  vector[N] vloss_pr;
  vector[N] tau_pr;
}

transformed parameters {
  // Subject-level parameters with Matt trick
  vector<lower=0,upper=1>[N] omega_0;
  vector<lower=0>[N] vwin;
  vector<lower=0>[N] vloss;
  vector<lower=0>[N] tau;

  omega_0 = Phi_approx(mu_pr[1] + sigma[1] * omega_0_pr);
  vwin = exp(mu_pr[2] + sigma[2] * vwin_pr);
  vloss = exp(mu_pr[3] + sigma[3] * vloss_pr);
  tau = exp(mu_pr[4] + sigma[4] * tau_pr);
}

model {
  // Prior
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 5);

  omega_0_pr ~ normal(0, 1);
  vwin_pr ~ normal(0, 1);
  vloss_pr ~ normal(0, 1);
  tau_pr ~ normal(0, 1);

  // Likelihood
  for (j in 1:N) {
    real omega = omega_0[j];
    for (k in 1:Tsubj[j]) {
      // Calculate likelihood with bernoulli distribution
      for (l in 1:L[j,k]){
        d[j, k, l] ~ bernoulli_logit(tau[j] * (omega * P - l));
      }
    if (explosion[j,k] ==0){
      omega = omega * (1 + vwin[j] * pumps[j,k] * inv(P));
    }
    else{
      omega = omega * (1 - vloss[j] * (1 - pumps[j,k] * inv(P)));
    }
      
    }
  }
}

generated quantities {
  // Actual group-level mean
  real<lower=0,upper=1> mu_omega_0 = Phi_approx(mu_pr[1]);
  real<lower=0> mu_vwin = exp(mu_pr[2]);
  real<lower=0> mu_vloss = exp(mu_pr[3]);
  real<lower=0> mu_tau = exp(mu_pr[4]);

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
      real omega = omega_0[j];

      log_lik[j] = 0;

      for (k in 1:Tsubj[j]) {
        for (l in 1:L[j,k]) {
          log_lik[j] += bernoulli_logit_lpmf(d[j, k, l] | tau[j] * (omega * P - l));
          y_pred[j, k, l] = bernoulli_logit_rng(tau[j] * (omega * P -l));
        }

        if (explosion[j,k] ==0){
          omega = omega * (1 + vwin[j] * pumps[j,k] * inv(P));
        }
        else{
          omega = omega * (1 - vloss[j] * (1 - pumps[j,k] * inv(P)));
        }
      }
    }
  }
}
