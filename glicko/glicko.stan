data {
  // training data for model fit
  int<lower=1> n_game;                               // # of unique games
  int<lower=1> n_period;                             // # of unique periods
  int<lower=1> n_player;                             // # of unique players
  int<lower=1> id_period[n_game];                    // period IDs
  int<lower=1> id_white[n_game];                     // white player IDs
  int<lower=1> id_black[n_game];                     // black player IDs
  int<lower=0, upper=1> score[n_game];               // game scores
  // testing data for posterior predictions
  int<lower=1> n_game_pred;                          // # of unique games (test)
  int<lower=1> id_white_pred[n_game_pred];           // white player IDs (test)
  int<lower=1> id_black_pred[n_game_pred];           // black player IDs (test)
}

parameters {
  matrix[n_period + 1, n_player] gamma;              // ratings
  real<lower=0> sigma_sq[n_period + 1, n_player];    // rating variances
  real<lower=0> tau_sq;                              // stochastic variance
  real beta;                                         // advantage of white
  real<lower=0, upper=1> rho;                        // autoregressive parameter
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

  int<lower=0, upper=1> score_rep[n_game];
  int<lower=0, upper=1> score_pred[n_game_pred];

  for (g in 1:n_game) {
    score_rep[g] = bernoulli_logit_rng(
      gamma[id_period[g] + 1, id_white[g]] - gamma[id_period[g] + 1, id_black[g]] + beta
    );
  }

  for (g in 1:n_game_pred) {
    score_pred[g] = bernoulli_logit_rng(
      gamma[n_period + 1, id_white_pred[g]] - gamma[n_period + 1, id_black_pred[g]] + beta
    );
  }

}
