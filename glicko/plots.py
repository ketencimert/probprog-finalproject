import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glicko.compute import get_empirical_score, get_synthetic_score

def plot_ppc(score_pp, observed_data, check, dim):

    fig_size = 5

    fig, axes = plt.subplots(1, len(check))

    fig.set_size_inches(len(check) * fig_size, fig_size)

    bins = [33, 28, 20, 20]

    for i in range(len(check)):
        main_title = 'T = {}'.format(check[i])

        empirical_score = get_empirical_score(observed_data, check[i].lower(), dim)

        sim_dist = get_synthetic_score(score_pp, observed_data, check[i].lower(), dim)

        axes[i].hist(sim_dist, bins=bins[i])

        axes[i].axvline(x=empirical_score, color='b', label='axvline - full height')

        axes[i].title.set_text(main_title)

        axes[i].set_ylabel("{}".format('Number of wins'))

        fig.suptitle('Posterior Predictive Checks w.r.t. Period')

    plt.show()

    return None


def plot_elbo(glicko_vi):
    for fname in glicko_vi.runset._stdout_files:
        with open(fname, "r") as f:
            text = f.read()

    text = text.split('\n')
    idx = text.index('Begin stochastic gradient ascent.')

    elbos = []
    iterations = []

    for i in range(idx + 2, len(text) - 4):
        cache = [x for x in text[i].split(" ") if x != ""]
        iterations.append(float(cache[0]))
        elbos.append(float(cache[1]))

    plt.title('Convergence of ELBO')
    plt.plot(iterations, elbos)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.show()

    return None

def plot_likelihood(glicko_map):

    for fname in glicko_map.runset._stdout_files:
        with open(fname, "r") as f:
            text = f.read()

        split = 'Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes '
        len_splitted = len(text.split(split))
        splitted = text.split(split)
        iterations = []
        loglikelihoods = []
        for i in range(1, len_splitted):
            cache = [x for x in splitted[i].strip().split(' ') if x != '']

            iterations.append(float(cache[0]))
            loglikelihoods.append(float(cache[1]))

        plt.title('Convergence of LogLikelihood')
        plt.plot(iterations, loglikelihoods)
        plt.xlabel('Iteration')
        plt.ylabel('LogLikelihood')
        plt.show()
    return None
