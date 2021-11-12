data {
  int<lower=1> n_game;                     // number of unique games
  int<lower=1> n_period;                   // number of unique periods
  int<lower=1> n_player;                   // number of unique players
  int<lower=1> id_period[n_game];          // period IDs
  int<lower=1> id_white[n_game];           // white player IDs
  int<lower=1> id_black[n_game];           // black player IDs
  int<lower=0, upper=1> score[n_game];     // game scores
}

parameters {
  matrix[n_period + 1, n_player] gamma;
  real<lower=0> sigma_sq[n_period + 1, n_player];
  real<lower=0> tau_sq;
  real beta;
  real<lower=0, upper=1> rho;
}

model {

  tau_sq ~ inv_gamma(4, 1.5);
  beta ~ normal(0, 5);

  sigma_sq[1] ~ inv_gamma(4, 2);
  for (t in 2:(n_period + 1)) {
    sigma_sq[t] ~ lognormal(log(sigma_sq[t - 1]), sqrt(tau_sq));
  }

  gamma[1] ~ normal(0, sqrt(sigma_sq[1]));
  for (t in 2:(n_period + 1)) {
    gamma[t] ~ normal(rho * gamma[t - 1], sqrt(sigma_sq[t]));
  }

  for (g in 1:n_game) {
    score[g] ~ bernoulli_logit(
      gamma[id_period[g] + 1, id_white[g]] - gamma[id_period[g] + 1, id_black[g]] + beta
    );
  }

}

generated quantities {
  real score_pp[n_game];

  for (g in 1:n_game) {
    score_pp[g] = bernoulli_logit_rng(
      gamma[id_period[g] + 1, id_white[g]] - gamma[id_period[g] + 1, id_black[g]] + beta
    );
  }
}