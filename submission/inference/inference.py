from collections import defaultdict

import arviz as az

import pandas as pd
import math
import numpy as np

from criticism.criticism import (
    plot_trace,
    plot_elbo,
    plot_loglikelihood,
    plot_bce
)

from utils import (
    pp_hmc,
    pp_vi,
    pp_map,
    pp_glickman,
    sigmoid
)


def evaluate(ratings_by_time_, chess_data):

    iteration_loss = []
    for match in chess_data.values:
        period = match[3]
        id_white = match[4]
        id_black = match[5]
        score = match[6]

        gamma_white = 0
        gamma_black = 0

        if period in ratings_by_time_.keys():

            if id_white in ratings_by_time_[period].keys():

                gamma_white = ratings_by_time_[period][id_white]

            if id_black in ratings_by_time_[period].keys():

                gamma_black = ratings_by_time_[period][id_black]

        p = sigmoid(gamma_white - gamma_black)
        iteration_loss.append(
            score * np.log(p) + (1 - score) * np.log(1 - p)
        )

    return np.mean(iteration_loss)


class Player:
    """
    Credits: https://github.com/deepy/glicko2
    """
    _tau = 0.5

    def getRating(self):
        return (self.__rating * 173.7178) + 1500

    def setRating(self, rating):
        self.__rating = (rating - 1500) / 173.7178

    rating = property(getRating, setRating)

    def getRd(self):
        return self.__rd * 173.7178

    def setRd(self, rd):
        self.__rd = rd / 173.7178

    rd = property(getRd, setRd)

    def __init__(self, rating=1500, rd=350, vol=0.06):
        # For testing purposes, preload the values
        # assigned to an unrated player.
        self.setRating(rating)
        self.setRd(rd)
        self.vol = vol

    def _preRatingRD(self):
        """ Calculates and updates the player's rating deviation for the
        beginning of a rating period.

        preRatingRD() -> None

        """
        self.__rd = math.sqrt(math.pow(self.__rd, 2) + math.pow(self.vol, 2))

    def update_player(self, rating_list, RD_list, outcome_list):
        """ Calculates the new rating and rating deviation of the player.

        update_player(list[int], list[int], list[bool]) -> None

        """
        # Convert the rating and rating deviation values for internal use.
        rating_list = [(x - 1500) / 173.7178 for x in rating_list]
        RD_list = [x / 173.7178 for x in RD_list]

        v = self._v(rating_list, RD_list)
        self.vol = self._newVol(rating_list, RD_list, outcome_list, v)
        self._preRatingRD()

        self.__rd = 1 / math.sqrt((1 / math.pow(self.__rd, 2)) + (1 / v))

        tempSum = 0
        for i in range(len(rating_list)):
            tempSum += self._g(RD_list[i]) * \
                       (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
        self.__rating += math.pow(self.__rd, 2) * tempSum

    # step 5
    def _newVol(self, rating_list, RD_list, outcome_list, v):
        """ Calculating the new volatility as per the Glicko2 system.

        Updated for Feb 22, 2012 revision. -Leo

        _newVol(list, list, list, float) -> float

        """
        # step 1
        a = math.log(self.vol ** 2)
        eps = 0.000001
        A = a

        # step 2
        B = None
        delta = self._delta(rating_list, RD_list, outcome_list, v)
        tau = self._tau
        if (delta ** 2) > ((self.__rd ** 2) + v):
            B = math.log(delta ** 2 - self.__rd ** 2 - v)
        else:
            k = 1
            while self._f(a - k * math.sqrt(tau ** 2), delta, v, a) < 0:
                k = k + 1
            B = a - k * math.sqrt(tau ** 2)

        # step 3
        fA = self._f(A, delta, v, a)
        fB = self._f(B, delta, v, a)

        # step 4
        while math.fabs(B - A) > eps:
            # a
            C = A + ((A - B) * fA) / (fB - fA)
            fC = self._f(C, delta, v, a)
            # b
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            # c
            B = C
            fB = fC

        # step 5
        return math.exp(A / 2)

    def _f(self, x, delta, v, a):
        ex = math.exp(x)
        num1 = ex * (delta ** 2 - self.__rating ** 2 - v - ex)
        denom1 = 2 * ((self.__rating ** 2 + v + ex) ** 2)
        return (num1 / denom1) - ((x - a) / (self._tau ** 2))

    def _delta(self, rating_list, RD_list, outcome_list, v):
        """ The delta function of the Glicko2 system.

        _delta(list, list, list) -> float

        """
        tempSum = 0
        for i in range(len(rating_list)):
            tempSum += self._g(RD_list[i]) * (
                    outcome_list[i] - self._E(rating_list[i], RD_list[i])
            )
        return v * tempSum

    def _v(self, rating_list, RD_list):
        """ The v function of the Glicko2 system.

        _v(list[int], list[int]) -> float

        """
        tempSum = 0
        for i in range(len(rating_list)):
            tempE = self._E(rating_list[i], RD_list[i])
            tempSum += math.pow(self._g(RD_list[i]), 2) * tempE * (1 - tempE)
        return 1 / tempSum

    def _E(self, p2rating, p2RD):
        """ The Glicko E function.

        _E(int) -> float

        """
        return 1 / (1 + math.exp(-1 * self._g(p2RD) * (
            self.__rating - p2rating)))

    def _g(self, RD):
        """ The Glicko2 g(RD) function.

        _g() -> float

        """
        return 1 / math.sqrt(1 + 3 * math.pow(RD, 2) / math.pow(math.pi, 2))

    def did_not_compete(self):
        """ Applies Step 6 of the algorithm. Use this for
        players who did not compete in the rating period.

        did_not_compete() -> None

        """
        self._preRatingRD()


def hmc(glicko_stan, observed_data, gamma_1=2, gamma_2=2):
    """
    Function to conduct inference
    :param glicko_stan: Path to Stan model file
    :param observed_data: Data dictionary to feed into Stan
    :param gamma_1: Trace plot parameter
    :param gamma_2: Trace plot parameter
    :return: Stan model, samples
    """
    glicko_mcmc = glicko_stan.sample(
        data=observed_data,
        seed=147,
        chains=4,
        parallel_chains=4,
        adapt_delta=0.95,
        refresh=500,
        iter_warmup=1000,
        iter_sampling=1000,
    )

    samples = az.from_cmdstanpy(
        posterior=glicko_mcmc,
    )

    plot_trace(samples, gamma_1, gamma_2)

    score_ppc_mcmc = pp_hmc(samples, observed_data)

    return glicko_mcmc, score_ppc_mcmc


def meanfield_vi(glicko_stan, observed_data):
    """
    Function to conduct inference
    :param glicko_stan: Path to Stan model file
    :param observed_data: Data dictionary to feed into Stan
    :return: Stan model, samples
    """
    glicko_vi = glicko_stan.variational(
        data=observed_data,
        seed=147,
        refresh=500,
        algorithm="meanfield",
        iter=10000,
        grad_samples=1,
        elbo_samples=100,
        adapt_engaged=True,
        tol_rel_obj=0.003,
        eval_elbo=100,
        adapt_iter=50,
        output_samples=1000
    )

    plot_elbo(glicko_vi)

    score_ppc_vi = pp_vi(glicko_vi, observed_data)

    return glicko_vi, score_ppc_vi


def map_opt(glicko_stan, observed_data):
    """
    Function to conduct inference
    :param glicko_stan: Path to Stan model file
    :param observed_data: Data dictionary to feed into Stan
    :return: Stan model, samples
    """
    glicko_map = glicko_stan.optimize(
        data=observed_data,
        seed=147,
        refresh=100,
        algorithm="lbfgs",
        init_alpha=0.001,
        iter=2000,
        tol_obj=1e-12,
        tol_rel_obj=1e4,
        tol_grad=1e-8,
        tol_rel_grad=1e7,
        tol_param=1e-8,
        history_size=5
    )
    plot_loglikelihood(glicko_map)
    score_ppc_map = pp_map(glicko_map, observed_data)

    return glicko_map, score_ppc_map


def glickman(observed_data_):
    """
    Function to conduct inference
    :param observed_data_: Data dictionary to feed into Stan
    :return: Samples, test probabilities
    """
    observed_data = dict()

    for (key, value) in observed_data_.items():

        if 'test' not in key:
            observed_data[key] = value

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

                black_games = all_games[
                    all_games['id_black'] == player].rename(
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

                    ratings_by_time_[period][player] =\
                        (rating - 1500) / 173.7178

                else:

                    score = games['score'].tolist()

                    ratings = [players[k].rating for k in games['id_black']]

                    rds = [players[k].rd for k in games['id_black']]

                    players[player].update_player(ratings, rds, score)

                    rating = players[player].rating

                    ratings_by_time.append(
                        (period, player, rating)
                    )

                    ratings_by_time_[period][player] =\
                        (rating - 1500) / 173.7178

                bce.append(-evaluate(ratings_by_time_, observed_data))

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

    max_period = max(ratings_by_time_.keys())

    id_white_test = observed_data_['id_white_test']
    id_black_test = observed_data_['id_black_test']

    probs = []

    for (id_white, id_black) in zip(id_white_test,
                                    id_black_test):
        gamma_white = ratings_by_time_[max_period][id_white]
        gamma_black = ratings_by_time_[max_period][id_black]

        probs += [sigmoid(gamma_white - gamma_black)]

    return score_ppc_glickman, probs
