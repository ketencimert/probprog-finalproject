from collections import defaultdict

import arviz as az
import bebi103

import pandas as pd

from inference.glicko2_inference import Player, evaluate

from criticism.criticism import (plot_trace,
                         plot_elbo,
                         plot_loglikelihood,
                         plot_bce
                         )

from utils import (
    pp_hmc,
    pp_vi,
    pp_map,
    pp_glickman
)


def hmc(glicko_stan, observed_data, gamma_1=2, gamma_2=2):
    with bebi103.stan.disable_logging():
        glicko_mcmc = glicko_stan.sample(
            data=observed_data,
            chains=4,
            iter_sampling=1000,
            adapt_delta=0.95,
            seed=123,
        )

    samples = az.from_cmdstanpy(
        posterior=glicko_mcmc,
    )

    plot_trace(samples, gamma_1, gamma_2)

    score_ppc_mcmc = pp_hmc(samples, observed_data)

    return glicko_mcmc, score_ppc_mcmc


def meanfield_vi(glicko_stan, observed_data):
    with bebi103.stan.disable_logging():
        glicko_vi = glicko_stan.variational(
            data=observed_data,
            algorithm="meanfield",
            output_samples=4000,
            grad_samples=5,
        )

    plot_elbo(glicko_vi)

    score_ppc_vi = pp_vi(glicko_vi, observed_data)

    return glicko_vi, score_ppc_vi


def map_opt(glicko_stan, observed_data):
    with bebi103.stan.disable_logging():
        glicko_map = glicko_stan.optimize(
            data=observed_data,
            algorithm="LBFGS",
            iter=2000,
        )

    plot_loglikelihood(glicko_map)

    score_ppc_map = pp_map(glicko_map, observed_data)

    return glicko_map, score_ppc_map


def glickman(observed_data):
    observed_data = pd.DataFrame.from_dict(
        observed_data
    )

    unique_periods = observed_data['id_period'].unique()

    unique_players = pd.concat([
        observed_data['id_white'],
        observed_data['id_black'],
    ]).unique()

    players = {
        k: Player() for k in unique_players
    }

    bce = [2, 1]
    ratings_by_time = []
    ratings_by_time_ = defaultdict(dict)

    while abs(bce[-2] - bce[-1]) > 1e-3:
        for period in unique_periods:

            all_games = observed_data[observed_data['id_period'] == period]

            for player in unique_players:

                white_games = all_games[all_games['id_white'] == player]

                black_games = all_games[all_games['id_black'] == player].rename(
                    columns={"id_white": "id_black",
                             "id_black": "id_white"})

                black_games['score'] = [not _ for _ in black_games['score']]

                games = pd.concat([white_games, black_games])

                if len(games) == 0:

                    players[player].did_not_compete()

                    rating = players[player].rating

                    ratings_by_time.append(
                        (period, player, rating)
                    )

                    ratings_by_time_[period][player] = (rating - 1500) / 173.7178

                else:

                    score = games['score'].tolist()

                    ratings = [players[k].rating for k in games['id_black']]

                    rds = [players[k].rd for k in games['id_black']]

                    players[player].update_player(ratings, rds, score)

                    rating = players[player].rating

                    ratings_by_time.append(
                        (period, player, rating)
                    )

                    ratings_by_time_[period][player] = (rating - 1500) / 173.7178

                bce.append(evaluate(ratings_by_time_, observed_data))

    bce.pop(0)

    bce.pop(0)

    ratings_by_time = pd.DataFrame(ratings_by_time).rename(
        columns={
            0: 'period',
            1: 'player',
            2: 'rating'
        }
    )

    ratings_by_time['rating'] = (ratings_by_time['rating'] - 1500) / 173.7178

    white_games = observed_data[observed_data['id_white'] == 3]

    black_games = observed_data[observed_data['id_black'] == 3].rename(
        columns={"id_white": "id_black",
                 "id_black": "id_white"})

    black_games['score'] = [not _ for _ in black_games['score']]

    games = pd.concat([white_games, black_games]).groupby('id_period').sum()

    iteration = list(range(len(bce)))

    delta = [bce[i + 1] - bce[i] for i in range(len(bce) - 1)]

    delta.insert(0, delta[0])

    plot_bce(iteration, bce, delta)

    score_ppc_glickman = pp_glickman(observed_data, ratings_by_time_)

    return score_ppc_glickman
