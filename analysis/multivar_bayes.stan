data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  vector[N] y;
  row_vector[K] x_pred;
}

parameters {
  vector[K] beta;
  real<lower=0> sigma;
}

model {
  beta ~ normal(0, 10);
  sigma ~ exponential(0.001);
  y ~ normal(X * beta, sigma);
}

generated quantities {
  real y_pred = normal_rng(dot_product(x_pred, beta), sigma);
  vector[N] y_rep;
  for (i in 1:N) {
    y_rep[i] = normal_rng(dot_product(row(X, i), beta), sigma);
  }

}
