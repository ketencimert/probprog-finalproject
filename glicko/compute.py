from collections import defaultdict
import json

import numpy as np
from numpy.random import binomial

import pandas as pd

from glicko.glicko2_inference import Player, sigmoid, evaluate


def bayesian_p(score_pp, observed_data, check, dim):
    p_values = dict()

    for i in range(len(check)):
        empirical_score = get_empirical_score(observed_data, check[i].lower(), dim)

        sim_dist = get_synthetic_score(score_pp, observed_data, check[i].lower(), dim)

        p_value = sim_dist > empirical_score

        p_value = np.sum(p_value / p_value.shape[0])

        p_values[check[i]] = p_value

    return p_values


def get_samples_vi(df_vi, glicko_vi, observed_data):
    df_vi.columns = glicko_vi.column_names

    keys = []

    for key in df_vi.keys():

        if 'score_pp' in key:
            keys.append(key)

    df_vi = df_vi[keys]

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_pp_vi = df_vi.rename(columns=new_names)

    return score_pp_vi


def get_empirical_score(observed_data, check, dim='period'):
    if dim == 'period':

        observed_dataframe = pd.DataFrame.from_dict(observed_data)
        observed_dataframe = observed_dataframe.groupby(
            'id_period'
        ).sum(
            'score'
        ).reset_index()[['id_period', 'score']]

    elif dim == 'player':

        observed_dataframe = pd.DataFrame.from_dict(observed_data)
        observed_dataframe = observed_dataframe.groupby(
            'id_white'
        ).sum(
            'score'
        ).reset_index()[['id_white', 'score']]

    if check == 'sum':

        empirical_score = observed_dataframe.values[:, 1].mean(0)

    elif check == 'std':

        empirical_score = observed_dataframe.values[:, 1].std(0)

    elif check == 'min':

        empirical_score = observed_dataframe.values[:, 1].min(0)

    elif check == 'max':

        empirical_score = observed_dataframe.values[:, 1].max(0)

    return empirical_score


def get_synthetic_score(score_pp, observed_data, check, dim='period'):
    if dim == 'period':

        sim_dist = pd.DataFrame(
            np.asarray(score_pp),
            columns=observed_data['id_period']
        ).T
        sim_dist = sim_dist.reset_index()
        sim_dist = sim_dist.rename(columns={'index': 'id_period'})

        name = 'id_period'

    elif dim == 'player':

        sim_dist = pd.DataFrame(
            np.asarray(score_pp),
            columns=observed_data['id_white']
        ).T
        sim_dist = sim_dist.reset_index()
        sim_dist = sim_dist.rename(columns={'index': 'id_white'})

        name = 'id_white'

    if check == 'sum':

        sim_dist = sim_dist.groupby(name).sum().mean(0)

    elif check == 'std':

        sim_dist = sim_dist.groupby(name).sum().std(0)

    elif check == 'min':

        sim_dist = sim_dist.groupby(name).sum().min(0)

    elif check == 'max':

        sim_dist = sim_dist.groupby(name).sum().max(0)

    return sim_dist


def get_samples_map(df_map, observed_data):
    keys = []

    for key in df_map.keys():

        if 'score_pp' in key:
            keys.append(key)

    df_map = df_map[keys]

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_pp_map = df_map.rename(columns=new_names)

    return score_pp_map


def run_glickman_inference(chess_data):
    chess_data = pd.DataFrame.from_dict(
        json.load(open(chess_data))
    )

    unique_periods = chess_data['id_period'].unique()

    unique_players = pd.concat([
        chess_data['id_white'],
        chess_data['id_black'],
    ]).unique()

    players = {
        k: Player() for k in unique_players
    }

    bce = [2, 1]
    ratings_by_time = []
    ratings_by_time_ = defaultdict(dict)

    while abs(bce[-2] - bce[-1]) > 1e-3:
        for period in unique_periods:

            all_games = chess_data[chess_data['id_period'] == period]

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

                bce.append(evaluate(ratings_by_time_, chess_data))

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

    white_games = chess_data[chess_data['id_white'] == 3]

    black_games = chess_data[chess_data['id_black'] == 3].rename(
        columns={"id_white": "id_black",
                 "id_black": "id_white"})

    black_games['score'] = [not _ for _ in black_games['score']]

    games = pd.concat([white_games, black_games]).groupby('id_period').sum()

    samples = []

    for match in chess_data.values:
        period = match[3]

        id_white = match[4]

        id_black = match[5]

        gamma_white = ratings_by_time_[period][id_white]

        gamma_black = ratings_by_time_[period][id_black]

        samples.append(binomial(n=1,
                                p=sigmoid(gamma_white - gamma_black),
                                size=4000
                                ))

    samples = np.asarray(samples)

    samples = pd.DataFrame(samples, index=chess_data['id_period']).T

    iteration = list(range(len(bce)))

    delta = [bce[i + 1] * bce[i] for i in range(len(bce) - 1)]

    delta.insert(0, abs(bce[0]))

    return samples, (iteration, bce, delta)

# def run_glickman_inference(chess_data):

#     chess_data = pd.DataFrame.from_dict(
#         json.load(open(chess_data))
#         )

#     unique_periods = chess_data['id_period'].unique()

#     unique_players = pd.concat([
#         chess_data['id_white'],
#         chess_data['id_black'],
#         ]).unique()

#     players = {
#         k:Player() for k in unique_players
#         }


#     bce = []
#     ratings_by_time = []
#     ratings_by_time_ = defaultdict(dict)

#     for period in unique_periods:

#         all_games = chess_data[chess_data['id_period'] == period]

#         for player in unique_players:

#             white_games = all_games[all_games['id_white'] == player]

#             black_games = all_games[all_games['id_black'] == player].rename(
#                     columns={"id_white": "id_black",
#                              "id_black": "id_white"})

#             black_games['score'] = [not _ for _ in black_games['score']]

#             games = pd.concat([white_games, black_games])

#             if len(games) == 0:

#                  players[player].did_not_compete()

#                  rating = players[player].rating

#                  ratings_by_time.append(
#                     (period, player, rating)
#                     )

#                  ratings_by_time_[period][player] = (rating - 1500) / 173.7178

#             else:

#                 score = games['score'].tolist()

#                 ratings = [players[k].rating for k in games['id_black']]

#                 rds = [players[k].rd for k in games['id_black']]

#                 players[player].update_player(ratings, rds, score)

#                 rating = players[player].rating

#                 ratings_by_time.append(
#                     (period, player, rating)
#                     )

#                 ratings_by_time_[period][player] = (rating - 1500) / 173.7178

#             bce.append(evaluate(ratings_by_time_, chess_data))

#     ratings_by_time = pd.DataFrame(ratings_by_time).rename(
#         columns={
#         0:'period',
#         1:'player',
#         2:'rating'
#         }
#         )

#     ratings_by_time['rating'] = (ratings_by_time['rating'] - 1500) / 173.7178

#     white_games = chess_data[chess_data['id_white'] == 3]

#     black_games = chess_data[chess_data['id_black'] == 3].rename(
#             columns={"id_white": "id_black",
#                      "id_black": "id_white"})

#     black_games['score'] = [not _ for _ in black_games['score']]

#     games = pd.concat([white_games, black_games]).groupby('id_period').sum()

#     samples = []

#     for match in chess_data.values:

#         period = match[3]

#         id_white = match[4]

#         id_black = match[5]

#         gamma_white = ratings_by_time_[period][id_white]

#         gamma_black = ratings_by_time_[period][id_black]

#         samples.append(binomial(n=1,
#                  p=sigmoid(gamma_white - gamma_black),
#                  size=4000
#                  ))

#     samples = np.asarray(samples)

#     samples = pd.DataFrame(samples, index=chess_data['id_period']).T

#     iteration = list(range(len(bce)))

#     delta = [bce[i+1] * bce[i] for i in range(len(bce)-1)]

#     delta.insert(0, abs(bce[0]))

#     plot_bce(iteration, bce, delta)

#     return samples
