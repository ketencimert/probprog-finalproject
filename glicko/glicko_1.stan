data {

    int<lower=1> gsize;  // total number of games

    int<lower=1> psize;  // total number of unique players

    int p1id[gsize]; //vector of ids for white players at current match id

    int p2id[gsize]; //vector of ids for black players at current match id

    int<lower=0, upper=gsize> score[gsize]; // vector of the scores for each game

}

parameters {

    vector[psize] gamma[gsize+1];  // array of player strength vectors by games the 1 is picked from standard normal

    vector<lower=0>[psize] sigma2[gsize+1];  // array of player variances

    real<lower=0> omega2;  // variance of player strength for t=1

    real<lower=0> tau;

    real beta;  // white player effect, not used right now

    real<lower=0, upper=1> rho;
}

model {

  // create initial vectors

    vector[psize] sigma[gsize+1];

    omega2 ~ inv_gamma(4, 2);

    tau ~ inv_gamma(4, 1.5);

    beta ~ normal(0, 25);

    for (p in 1:psize)

        gamma[1][p] ~ normal(0, sqrt(omega2));

    for (p in 1:psize)

        sigma2[1][p] ~  lognormal(log(omega2), tau);

    for (g in 1:gsize)
        for (p in 1:psize)

            sigma2[g+1][p] ~ lognormal(log(sigma2[g][p]), tau);

    for (g in 1:gsize+1)
        for (p in 1:psize)

            sigma[g][p] = sqrt(sigma2[g][p]);

    for (g in 1:gsize)
        for (p in 1:psize)

            gamma[g+1][p] ~ normal(rho * gamma[g][p], sigma[g][p]);

    for (g in 1:gsize)

        score[g] ~ bernoulli_logit(gamma[g+1][p1id[g]] - gamma[g+1][p2id[g]]+beta);

}
