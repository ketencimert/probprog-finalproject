import numpy as np
from numpy.random import binomial

import pandas as pd

from glicko.glicko2_inference import sigmoid


def pp_hmc(samples, observed_data):
    score_pp_mcmc = np.concatenate(
        [
            samples.posterior[
                "score_pp"][chain] for chain in range(4)
        ]
    )

    score_pp_mcmc = pd.DataFrame(score_pp_mcmc)

    keys = []

    for key in score_pp_mcmc.keys():
        keys.append(key)

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_pp_mcmc = score_pp_mcmc.rename(columns=new_names)

    return score_pp_mcmc


def pp_vi(glicko_vi, observed_data):
    df_vi = glicko_vi.variational_sample

    df_vi.columns = glicko_vi.column_names

    keys = []

    for key in df_vi.keys():

        if 'score_pp' in key:
            keys.append(key)

    df_vi = df_vi[keys]

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_pp_vi = df_vi.rename(columns=new_names)

    return score_pp_vi


def pp_map(glicko_map, observed_data):
    df_map = glicko_map.optimized_params_pd

    keys = []

    for key in df_map.keys():

        if 'score_pp' in key:
            keys.append(key)

    df_map = df_map[keys]

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_pp_map = df_map.rename(columns=new_names)

    return score_pp_map


def pp_glickman(observed_data, ratings_by_time_):
    samples = []

    for match in observed_data.values:
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

    samples = pd.DataFrame(samples, index=observed_data['id_period']).T
    return samples