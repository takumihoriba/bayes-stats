data {
  int<lower=0> N; // number of observations
  vector[N] xs;   // independent variable
  vector[N] ys;   // dependent variable
  real x_pred;
}

parameters {
  real slope;
  real<lower=0> sigma;
}

model {
  // prior
  slope     ~ student_t(3, 0, 100);
  sigma     ~ exponential(0.001);

  // likelihood
  ys ~ normal(slope*xs, sigma);
}

generated quantities {
  real y_pred = normal_rng(slope * x_pred, sigma);
}
