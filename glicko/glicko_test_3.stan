data {

    int<lower=1> gsize;  // total number of games

    int<lower=1> psize;  // total number of unique players

    int p1id[gsize]; //vector of ids for white players at current match id

    int p2id[gsize]; //vector of ids for black players at current match id

    int<lower=0, upper=gsize> score[gsize]; // vector of the scores for each game

}

parameters {

    real beta;

    real alpha;

    real<lower=0> omega2;

    real<lower=0> tau;

    real<lower=0, upper=1> rho;

    matrix[gsize+1, psize] gamma;

    matrix[gsize+1, psize] logsigma2;

}

model {

    matrix[gsize+1, psize] sigma;

    omega2 ~ inv_gamma(4, 2);

    tau ~ inv_gamma(4, 1.5);

    beta ~ normal(0, 25);

    gamma[1] ~ normal(0, sqrt(omega2));

    logsigma2[1] ~  normal(log(omega2), tau);

    for (g in 1:gsize)

        logsigma2[g+1] ~ normal(logsigma2[g], tau);

    sigma = sqrt(exp(logsigma2));

    for (g in 1:gsize)

        gamma[g+1] ~ normal(rho * gamma[g], sigma[g]);

    for (g in 1:gsize)

        score[g] ~ bernoulli_logit(gamma[g+1][p1id[g]] - gamma[g+1][p2id[g]] + beta);

}