# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:18:20 2021

@author: Mert
"""
import sys
import argparse

from collections import defaultdict

import pandas as pd
import numpy as np

import json

#ToDo: Combine the periods.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_size', default=100, type=int)

    parser.add_argument('--min_game', default=10, type=int)

    parser.add_argument('--cut', default=400, type=int)

    parser.add_argument('--periods', default=20, type=int)

    parser.add_argument('--combine_every', default=12, type=int)

    args = parser.parse_args()

    players_rename = dict()

    period_rename = dict()

    data = pd.read_csv('./primary_training_part1.csv')

    data = data[data['MonthID'] <= args.periods]

    data = data[data['WhiteScore'] != 0.5]

    white_group = data.groupby('WhitePlayer').count()

    black_group = data.groupby('BlackPlayer').count()

    groups = pd.concat([white_group, black_group]).sort_values(
        'PTID',
        ascending=False,
        )[:args.cut].index.tolist()

    data = data[data['WhitePlayer'].isin(groups)]

    data = data[data['BlackPlayer'].isin(groups)]

    white_group = data.groupby('WhitePlayer').count()

    white_group = white_group[white_group['PTID']>args.min_game].index.tolist()

    black_group = data.groupby('BlackPlayer').count()

    black_group = black_group[black_group['PTID']>args.min_game].index.tolist()

    data = data[data['WhitePlayer'].isin(white_group)]

    data = data[data['BlackPlayer'].isin(black_group)]

    n_game = data.shape[0]

    id_period = data['MonthID']

    n_period = len(data['MonthID'].unique())

    id_white = data['WhitePlayer']

    id_black = data['BlackPlayer']

    score = data['WhiteScore']

    players = pd.Series(np.concatenate(
        [
         id_white.values,
         id_black.values,
         ]
        )).unique()

    i = 1

    for player in players:

        players_rename[player] = i

        i += 1

    id_white = id_white.values.tolist()

    id_black = id_black.values.tolist()

    for i in range(len(id_white)):

        id_white[i] = int(players_rename[id_white[i]])

    for i in range(len(id_black)):

        id_black[i] = int(players_rename[id_black[i]])

    #do mapping here
    i = 1

    k = 0

    for period in id_period.unique():

        if k == args.combine_every:

            k = 0

        if k != 0:

            period_rename[period] = period_rename[period-1]

        else:

            period_rename[period] = i

            i += 1
        
        k += 1
            
    id_period = id_period.values.tolist()

    for i in range(len(id_period)):

            id_period[i] = int(period_rename[id_period[i]])

    data = {
        'n_game': int(n_game),
        'n_period': int(n_period),
        'n_player': len(players_rename),
        'id_period': id_period,
        'id_white': id_white,
        'id_black': id_black,
        'score': [int(s) for s in score],
        }

    with open('chess.data.json', 'w') as fp:

        json.dump(data, fp)