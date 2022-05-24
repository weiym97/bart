
data {
  int<lower=1> N;            // Number of subjects
  int<lower=1> T;            // Maximum number of trials
  int<lower=0> Tsubj[N];     // Number of trials for each subject
  int<lower=2> P;            // Number of max pump + 1 ** CAUTION **
  int<lower=0> L[N, T];      // Number of decision times
  int<lower=0> pumps[N, T];  // Number of pump
  int<lower=0,upper=1> explosion[N, T];  // Whether the balloon exploded (0 or 1)
  real r[P-1];
  real r_accu[P];
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
  vector[N] phi_pr;
  vector[N] eta_pr;
  vector[N] gam_pr;
  vector[N] tau_pr;
}

transformed parameters {
  // Subject-level parameters with Matt trick
  vector<lower=0,upper=1>[N] phi;
  vector<lower=0>[N] eta;
  vector<lower=0>[N] gam;
  vector<lower=0>[N] tau;

  phi = Phi_approx(mu_pr[1] + sigma[1] * phi_pr);
  eta = exp(mu_pr[2] + sigma[2] * eta_pr);
  gam = exp(mu_pr[3] + sigma[3] * gam_pr);
  tau = exp(mu_pr[4] + sigma[4] * tau_pr);
}

model {
  // Prior
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  phi_pr ~ normal(0, 1);
  eta_pr ~ normal(0, 1);
  gam_pr ~ normal(0, 1);
  tau_pr ~ normal(0, 1);

  // Likelihood
  for (j in 1:N) {
    vector[P] n_succ;
    vector[P] n_pump;
    vector[P] Pre_prob_pump;
    for (i in 1:P){
      n_succ[i] = 0;
      n_pump[i] = 0;
      Pre_prob_pump[i] = 0;
    }

    for (k in 1:Tsubj[j]) {
      real p_no_burst_accu;
      real max_u;
      real p_burst;
      real u;
      real omega;    // Optimal number of pumps
      
      
      //ATTENTION: Revised by swimming, original code was wrong
      
      
      max_u=0;
      omega=0;
      p_no_burst_accu=1;
      for (t in 1:L[j,k]){
        p_burst = 1 - ((phi[j] + eta[j] * n_succ[t]) / (1 + eta[j] * n_pump[t]));
        p_no_burst_accu=p_no_burst_accu*(1-p_burst);
        u=p_no_burst_accu*r_accu[t]^gam[j];
        if (u>max_u){
          max_u=u;
          omega=t;
        }
      }

      // Calculate likelihood with bernoulli distribution
      for (l in 1:L[j,k]){
        d[j, k, l] ~ bernoulli_logit(tau[j] * (omega - l));
        //update n_succ and n_pump and probability of pumps after each trail ends
        if ((l<L[j,k]) || (explosion[j,k]==0))
          n_succ[l] =n_succ[l]+1;
        n_pump[l] = n_pump[l] +1;
        Pre_prob_pump[l] = (n_pump[l] - n_succ[l]) /n_pump[l];
      }
    }
  }
}

generated quantities {
  // Actual group-level mean
  real<lower=0, upper=1> mu_phi = Phi_approx(mu_pr[1]);
  real<lower=0> mu_eta = exp(mu_pr[2]);
  real<lower=0> mu_gam = exp(mu_pr[3]);
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
      vector[P] n_succ;
      vector[P] n_pump;
      vector[P] Pre_prob_pump;
      for (i in 1:P){
        n_succ[i] = 0;
        n_pump[i] = 0;
        Pre_prob_pump[i] = 0;
      }

      log_lik[j] = 0;

      for (k in 1:Tsubj[j]) {
        real p_no_burst_accu;
        real max_u;
        real p_burst;
        real u;
        real omega;    // Optimal number of pumps
      
      
        //ATTENTION: Revised by swimming, original code was wrong
      
      
        max_u=0;
        omega=0;
        p_no_burst_accu=1;
        for (t in 1:L[j,k]){
          p_burst = 1 - ((phi[j] + eta[j] * n_succ[t]) / (1 + eta[j] * n_pump[t]));
          p_no_burst_accu=p_no_burst_accu*(1-p_burst);
          u=p_no_burst_accu*r_accu[t]^gam[j];
          if (u>max_u){
            max_u=u;
            omega=t;
          }
        }

        for (l in 1:L[j,k]) {
          log_lik[j] += bernoulli_logit_lpmf(d[j, k, l] | tau[j] * (omega - l));
          y_pred[j, k, l] = bernoulli_logit_rng(tau[j] * (omega - l));
          //update n_succ and n_pump and probability of pumps after each trail ends
          if ((l<L[j,k]) || (explosion[j,k]==0))
            n_succ[l] =n_succ[l]+1;
          n_pump[l] = n_pump[l] +1;
          Pre_prob_pump[l] = (n_pump[l] - n_succ[l]) /n_pump[l];
          }
      }
    }
  }
}

