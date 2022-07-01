
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
  vector[7] mu_pr;
  vector<lower=0>[7] sigma;

  // Normally distributed error for Matt trick
  vector[N] psi_pr;
  vector[N] xi_pr;
  vector[N] gamma_pr;
  vector[N] tau_pr;
  vector[N] lambda_pr;
  vector[N] alpha_pr;
  vector[N] beta_pr;
}

transformed parameters {
  // Subject-level parameters with Matt trick
  vector<lower=0,upper=1>[N] psi;
  vector<lower=0>[N] xi;
  vector<lower=0>[N] gamma;
  vector<lower=0>[N] tau;
  vector<lower=0>[N] lambda;
  vector<lower=0,upper=1>[N] alpha;
  vector<lower=0>[N] beta;

  psi = Phi_approx(mu_pr[1] + sigma[1] * psi_pr);
  xi = exp(mu_pr[2] + sigma[2] * xi_pr);
  gamma = exp(mu_pr[3] + sigma[3] * gamma_pr);
  tau = exp(mu_pr[4] + sigma[4] * tau_pr);
  lambda = exp(mu_pr[5] + sigma[5] * lambda_pr);
  alpha = Phi_approx(mu_pr[6] + sigma[6] * alpha_pr);
  beta = exp(mu_pr[7] +  sigma[7] * beta_pr);
}

model {
  // Prior
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 5);

  psi_pr ~ normal(0, 1);
  xi_pr ~ normal(0, 1);
  gamma_pr ~ normal(0, 1);
  tau_pr ~ normal(0, 1);
  lambda_pr ~ normal(0,1);
  alpha_pr ~ normal(0,1);
  beta_pr ~ normal(0,1);
  
  


  // Likelihood
  for (j in 1:N) {
    // Initialize n_succ and n_pump for a subject
    int n_succ = 0;  // Number of successful pumps
    int n_pump = 0;  // Number of total pumps
    
    real A = 0.04355644;
    real B = -0.0988012;
    real C = 0.02832168;
    real RPE = 0;

    for (k in 1:Tsubj[j]) {
      real p_burst;  // Belief on a balloon to be burst
      real omega;    // Optimal number of pumps
      real temp_0;
      real temp_1;
      real temp_2;
      

      p_burst = exp(-xi[j] * n_pump) * psi[j] + (1 - exp(-xi[j] * n_pump)) * ((n_pump - n_succ) / (n_pump + 1e-5));
      temp_0 = C * lambda[j] * (1 + beta[j] * RPE) * p_burst - C * log1m(p_burst) - B * gamma[j];
      temp_1 = B * lambda[j] * (1 + beta[j] * RPE) * p_burst - 2 * A * gamma[j] - B * log1m(p_burst);
      temp_2 = A * lambda[j] * (1 + beta[j] * RPE) * p_burst- A * log1m(p_burst);
      omega = (- temp_1 + sqrt(temp_1 * temp_1 - 4 * temp_0 * temp_2)) / (2 * temp_2);
      
      

      // Calculate likelihood with bernoulli distribution
      for (l in 1:L[j,k]){
        d[j, k, l] ~ bernoulli_logit(tau[j] * (omega - l));
      }
      // Update n_succ and n_pump after each trial ends
      n_succ += pumps[j, k] - explosion[j, k];
      n_pump += pumps[j, k];
      if (r_accu[pumps[j,k] + 1] - (A * (n_pump * inv(k)) ^ 2 + B * n_pump * inv(k) + C)>0){
          RPE += alpha[j] * ((r_accu[pumps[j,k] + 1] - (A * (n_pump * inv(k)) ^ 2 + B * n_pump * inv(k) + C)) * (1 - explosion[j,k])- RPE);
        }
      else{
          RPE -= alpha[j] * RPE;
        }
    }
  }
}

generated quantities {
  // Actual group-level mean
  real<lower=0, upper=1> mu_psi = Phi_approx(mu_pr[1]);
  real<lower=0> mu_xi = exp(mu_pr[2]);
  real<lower=0> mu_gamma = exp(mu_pr[3]);
  real<lower=0> mu_tau = exp(mu_pr[4]);
  real<lower=0> mu_lambda = exp(mu_pr[5]);
  real<lower=0> mu_alpha = exp(mu_pr[6]);
  real<lower=0> mu_beta = exp(mu_pr[7]);

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
      real A = 0.04355644;
      real B = -0.0988012;
      real C = 0.02832168;
      real RPE = 0;

      log_lik[j] = 0;

      for (k in 1:Tsubj[j]) {
        real p_burst;  // Belief on a balloon to be burst
        real omega;    // Optimal number of pumps
        real temp_0;
        real temp_1;
        real temp_2;
       

        p_burst = exp(-xi[j] * n_pump) * psi[j] + (1 - exp(-xi[j] * n_pump)) * ((n_pump - n_succ) / (n_pump + 1e-5));
        temp_0 = C * lambda[j] * (1 + beta[j] * RPE) * p_burst - C * log1m(p_burst) - B * gamma[j];
        temp_1 = B * lambda[j] * (1 + beta[j] * RPE) * p_burst - 2 * A * gamma[j] - B * log1m(p_burst);
        temp_2 = A * lambda[j] * (1 + beta[j] * RPE) * p_burst- A * log1m(p_burst);
        omega = (- temp_1 + sqrt(temp_1 * temp_1 - 4 * temp_0 * temp_2)) / (2 * temp_2);

        

        for (l in 1:L[j,k]) {
          log_lik[j] += bernoulli_logit_lpmf(d[j, k, l] | tau[j] * (omega - l));
          y_pred[j, k, l] = bernoulli_logit_rng(tau[j] * (omega - l));
        }

        n_succ += pumps[j, k] - explosion[j, k];
        n_pump += pumps[j, k];
        if (r_accu[pumps[j,k] + 1] - (A * (n_pump * inv(k)) ^ 2 + B * n_pump * inv(k) + C)>0){
          RPE += alpha[j] * ((r_accu[pumps[j,k] + 1] - (A * (n_pump * inv(k)) ^ 2 + B * n_pump * inv(k) + C)) * (1 - explosion[j,k])- RPE);
        }
        else{
          RPE -= alpha[j] * RPE;
        }
      }
    }
  }
}

