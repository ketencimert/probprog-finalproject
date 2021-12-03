import math
import numpy as np
from numpy.random import binomial

import pandas as pd


def sigmoid(x):
    """
    Function to compute sigmoid
    :param x: logit value
    return: probability
    """
    return 1 / (1 + math.exp(-x))


def bayesian_p(score_ppc, observed_data, check, dim):
    """
    Function tocompute Bayesian p-value
    :param score_ppc: Samples
    :param observed_data: Real data
    :param check: PP Check
    :param dim: Marginalization dim
    return: Bayesian p-values
    """
    p_values = dict()

    for i in range(len(check)):
        empirical_score = get_empirical_score(
            observed_data,
            check[i].lower(),
            dim
            )

        sim_dist = get_synthetic_score(
            score_ppc,
            observed_data,
            check[i].lower(),
            dim
            )

        p_value = sim_dist > empirical_score

        p_value = np.sum(p_value / p_value.shape[0])

        p_values[check[i]] = p_value

    return p_values


def get_empirical_score(observed_data, check, dim='period'):
    """
    Function to compute empirical statistics
    :param observed_data: Real data
    :param check: PP Check
    :param dim: Marginalization dim
    return: Empirical statistics
    """
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


def get_synthetic_score(score_ppc, observed_data, check, dim='period'):
    """
    Function to compute empirical statistics
    :param score_ppc: Synthetic data
    :param observed_data: Real data
    :param check: PP Check
    :param dim: Marginalization dim
    return: Synthetic statistics
    """
    if dim == 'period':

        sim_dist = pd.DataFrame(
            np.asarray(score_ppc),
            columns=observed_data['id_period']
        ).T
        sim_dist = sim_dist.reset_index()
        sim_dist = sim_dist.rename(columns={'index': 'id_period'})

        name = 'id_period'

    elif dim == 'player':

        sim_dist = pd.DataFrame(
            np.asarray(score_ppc),
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


def pp_hmc(samples, observed_data):
    """
    Function to generate samples
    :param samples: MAP inference cmdstanpy dataframe
    :param observed_data: Real data
    return: Samples
    """
    score_ppc_mcmc = np.concatenate(
        [
            samples.posterior[
                "score_ppc"][chain] for chain in range(4)
        ]
    )

    score_ppc_mcmc = pd.DataFrame(score_ppc_mcmc)

    keys = []

    for key in score_ppc_mcmc.keys():
        keys.append(key)

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_ppc_mcmc = score_ppc_mcmc.rename(columns=new_names)

    return score_ppc_mcmc


def pp_vi(glicko_vi, observed_data):
    """
    Function to generate samples
    :param glicko_vi: Cmdstanpy VI method
    :param observed_data: Real data
    return: Samples
    """
    df_vi = glicko_vi.variational_sample

    df_vi.columns = glicko_vi.column_names

    keys = []

    for key in df_vi.keys():

        if 'score_ppc' in key:
            keys.append(key)

    df_vi = df_vi[keys]

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_ppc_vi = df_vi.rename(columns=new_names)

    return score_ppc_vi


def pp_map(glicko_map, observed_data):
    """
    Function to generate samples
    :param glicko_map: Cmdstanpy MAP method
    :param observed_data: Real data
    return: Samples
    """
    df_map = glicko_map.optimized_params_pd

    keys = []

    for key in df_map.keys():

        if 'score_ppc' in key:
            keys.append(key)

    df_map = df_map[keys]

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_ppc_map = df_map.rename(columns=new_names)

    return score_ppc_map


def pp_glickman(observed_data, ratings_by_time_):
    """
    Function to generate samples
    :param observed_data: Real data
    :param ratings_by_time_:  Dictionary - (period, player)
    return: Samples
    """
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
