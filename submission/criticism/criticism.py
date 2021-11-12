import matplotlib.pyplot as plt
import numpy as np

from utils import (
    get_empirical_score,
    get_synthetic_score,
    bayesian_p
)

def plot_ppc(score_ppc, observed_data, check, dim):
    fig_size = 5

    fig, axes = plt.subplots(1, len(check))

    fig.set_size_inches(len(check) * fig_size, fig_size)

    bins = [33, 28, 20, 20]

    p_values = bayesian_p(score_ppc, observed_data, check, dim)

    for i in range(len(check)):
        main_title = 'T = {} (p-value : {:.2f})'.format(check[i], p_values[check[i]])

        empirical_score = get_empirical_score(observed_data, check[i].lower(), dim)

        sim_dist = get_synthetic_score(score_ppc, observed_data, check[i].lower(), dim)

        axes[i].hist(sim_dist, bins=bins[i])

        axes[i].axvline(x=empirical_score, color='r')

        axes[i].title.set_text(main_title)

        axes[i].set_ylabel("{}".format('Number of wins'))

        fig.suptitle('Posterior Predictive Checks w.r.t. Period')

    plt.show()

    return None


def plot_trace(samples, gamma_1, gamma_2):
    draws = np.asarray(samples.posterior['gamma'])[:, :, gamma_1, gamma_2]
    iteration = list(range(draws.shape[1]))
    chains = dict()

    plt.figure(figsize=(10, 5))

    for i in range(draws.shape[0]):
        chains[i] = draws[i, :]

    for key in chains.keys():
        plt.plot(iteration, chains[key], label='Chain {}'.format(key))

    plt.legend()

    return None


def plot_elbo(glicko_vi):
    for fname in glicko_vi.runset._stdout_files:
        with open(fname, "r") as f:
            text = f.read()

    text = text.split('\n')
    idx = text.index('Begin stochastic gradient ascent.')

    elbos = []
    deltas = []
    iterations = []

    for i in range(idx + 2, len(text) - 4):
        cache = [x for x in text[i].split(" ") if x != ""]
        iterations.append(float(cache[0]))
        elbos.append(float(cache[1]))
        deltas.append(float(cache[2]))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(7, 4)

    color = 'tab:red'

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('ELBO', color=color)

    ax1.plot(iterations, elbos, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Delta', color=color)
    ax2.plot(iterations, deltas, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title('Convergence of ELBO')
    plt.show()

    return None


def plot_loglikelihood(glicko_map):
    for fname in glicko_map.runset._stdout_files:
        with open(fname, "r") as f:
            text = f.read()

        split = 'Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes '
        len_splitted = len(text.split(split))
        splitted = text.split(split)
        iterations = []
        loglikelihoods = []
        deltas = []
        for i in range(1, len_splitted):
            cache = [x for x in splitted[i].strip().split(' ') if x != '']

            iterations.append(float(cache[0]))
            loglikelihoods.append(float(cache[1]))
            deltas.append(float(cache[2]))

        fig, ax1 = plt.subplots()
        fig.set_size_inches(7, 4)

        color = 'tab:red'

        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('LogLikelihood', color=color)

        ax1.plot(iterations, loglikelihoods, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Delta', color=color)
        ax2.plot(iterations, deltas, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.title('Convergence of LogLikelihood')
        plt.show()

        return None


def plot_bce(iteration, bce, delta):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(7, 4)

    color = 'tab:red'

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('BCE', color=color)

    ax1.plot(iteration, bce, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Delta', color=color)
    ax2.plot(iteration, delta, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title('Convergence of BCE')
    plt.show()

    return None