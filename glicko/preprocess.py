# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:18:20 2021

@author: Mert

Download data from: https://www.kaggle.com/c/ChessRatings2

primary_training_part1.zip

"""

import argparse

from collections import defaultdict

import pandas as pd
import numpy as np

import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_size', default=100, type=int)

    args = parser.parse_args()

    players_rename = dict()

    data = pd.read_csv('./primary_training_part1.csv')[:args.sample_size] #stan memory issues...

    match_id = data['PTID'] #mid

    player1_id = data['WhitePlayer'] #p1id

    player2_id = data['BlackPlayer'] #p2id

    white_score = data['WhiteScore'] #score

    game_size = data['PTID'].unique().size #gsize

    players = pd.Series(np.concatenate(
        [
         player1_id.values,
         player2_id.values,
         ]
        )).unique()

    i = 1

    for player in players:

        players_rename[player] = i

        i += 1

    player1_id = player1_id.values.tolist()
    player2_id = player2_id.values.tolist()

    for i in range(len(player1_id)):

        player1_id[i] = players_rename[player1_id[i]]

    for i in range(len(player2_id)):

        player2_id[i] = players_rename[player2_id[i]]

    data = {
        'mid': match_id.values.tolist(),
        'p1id': player1_id,
        'p2id': player2_id,
        'score': (white_score==1.0).astype(int).tolist(), #how to deal with all square
        'gsize': game_size,
        'psize': len(players_rename),
        }

    with open('chess.data.json', 'w') as fp:
        json.dump(data, fp)