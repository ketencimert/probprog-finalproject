import numpy as np
import pandas as pd


def bayesian_p(score_pp, observed_data, check, dim):
    p_values = dict()

    for i in range(len(check)):
        main_title = 'T = {}'.format(check[i])

        empirical_score = get_empirical_score(observed_data, check[i].lower(), dim)

        sim_dist = get_synthetic_score(score_pp, observed_data, check[i].lower(), dim)

        p_value = sim_dist > empirical_score

        p_value = np.sum(p_value / p_value.shape[0])

        p_values[check[i]] = p_value

    return p_values

def get_samples_vi(df_vi, glicko_vi):

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

def get_samples_map(df_map):

    keys = []

    for key in df_map.keys():

        if 'score_pp' in key:
            keys.append(key)

    df_map = df_map[keys]

    new_names = {k: v for (k, v) in zip(keys, observed_data['id_period'])}

    score_pp_map = df_map.rename(columns=new_names)

    return score_pp_map
