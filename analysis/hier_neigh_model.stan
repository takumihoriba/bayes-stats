data {
  int<lower=1> N;                  // Number of observations
  int<lower=1> K;                  // Number of predictors
  matrix[N, K] X;                  // Design matrix
  vector[N] y;                     // Outcome variable
  int<lower=1> G;                  // Number of neighborhoods
  int<lower=1, upper=G> gid[N];    // Neighborhood ID for each obs
  row_vector[K] x_pred;           // New observation for prediction
}

parameters {
  vector[K] beta;                 // Coefficients for predictors
  real mu_alpha;                 // Hyper mean for intercepts
  real<lower=0> sigma;           // Observation noise
  real<lower=0> sigma_alpha;     // Variation between neighborhoods
  vector[G] alpha;               // Varying intercepts for neighborhoods
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
  y_pred = normal_rng(alpha[gid[1]] + x_pred * beta, sigma);
}
