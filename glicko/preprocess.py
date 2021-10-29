# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:18:20 2021

@author: Mert
"""

import argparse

from collections import defaultdict

import pandas as pd
import numpy as np

import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_size', default=100, type=int)
    
    parser.add_argument('--periods', default=2, type=int)

    args = parser.parse_args()

    players_rename = dict()

    data = pd.read_csv('./primary_training_part1.csv')

    data = data[data['MonthID'] <= args.periods]
    
    data = data[data['WhiteScore'] != 0.5]
    
    n_game = data.shape[0]
    
    n_period = max(
        pd.concat([
            data['WhitePlayer'],  data['BlackPlayer']
            ])
        )
    
    id_period = data['MonthID']
    
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

        id_white[i] = players_rename[id_white[i]]

    for i in range(len(id_black)):

        id_black[i] = players_rename[id_black[i]]

    data = {
        'n_game': int(n_game),
        'n_period': int(n_period),
        'id_period': [int(p) for p in id_period],
        'id_white': [int(w) for w in id_white],
        'id_black': [int(b) for b in id_black],
        'score': [int(s) for s in score],
        }

    with open('chess.data.json', 'w') as fp:
        
        json.dump(data, fp)