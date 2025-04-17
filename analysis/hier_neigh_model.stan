data {
  int<lower=1> N;                  // Number of observations
  int<lower=1> K;                  // Number of predictors
  matrix[N, K] X;                  // Design matrix
  vector[N] y;                     // Outcome variable
  int<lower=1> G;                  // Number of neighborhoods
  int<lower=1, upper=G> gid[N];    // Neighborhood ID for each obs
  row_vector[K] x_pred; 
  int<lower=1, upper=G> group_pred;
}

parameters {
  vector[K] beta;
  real mu_alpha;
  real<lower=0> sigma;    
  real<lower=0> sigma_alpha;
  vector[G] alpha;
}

model {
  // Priors
  beta ~ normal(0, 10);
  mu_alpha ~ normal(0, 100);
  sigma ~ exponential(0.001);
  sigma_alpha ~ exponential(0.001);
  alpha ~ normal(mu_alpha, sigma_alpha);

  // Likelihood
  y ~ normal(alpha[gid] + X * beta, sigma);
}

generated quantities {
  real y_pred;
  vector[N] y_rep;
  y_pred = normal_rng(alpha[group_pred] + x_pred * beta, sigma);
  for (i in 1:N) {
    y_rep[i] = normal_rng(alpha[gid[i]] + X[i] * beta, sigma);
  }
}
