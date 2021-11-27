from cmdstanpy.stanfit import CmdStanMCMC, CmdStanVB, CmdStanMLE
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import (
    get_empirical_score,
    get_synthetic_score,
    bayesian_p
)


def get_posterior_predictions(model, quantity):
    """
    Function to compute binomial cross entropy
    :param model: Stan model object
    :param quantity: Name of generated quantity
    :return: Maximum a posteriori probability estimates
    """
    if isinstance(model, CmdStanMCMC):
        df = model.draws_pd()
        y_pred = df[
            [col for col in df if col.startswith(quantity)]
        ].mean(axis=0)
    elif isinstance(model, CmdStanVB):
        df = model.variational_sample
        df.columns = model.column_names
        y_pred = df[
            [col for col in df if col.startswith(quantity)]
        ].mean(axis=0)
    elif isinstance(model, CmdStanMLE):
        df = model.optimized_params_pd
        y_pred = df[
            [col for col in df if col.startswith(quantity)]
        ].mean(axis=0)
    else:
        raise TypeError("CmdStan model must be MCMC, VB, or MLE")
    return y_pred


def get_binary_cross_entropy(y, y_pred, eta=0.01):
    """
    Function to compute binomial cross entropy
    :param y: True labels
    :param y_pred: Predicted probabilities
    :param eta: Clipping threshold
    :return: Binary cross entropy
    """
    y = np.array(y)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, eta, 1 - eta)
    binomial_deviance = - np.mean(
        y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )
    return binomial_deviance


def get_misclassification_error(y, y_pred):
    """
    Function to compute binomial cross entropy
    :param y: True labels
    :param y_pred: Predicted probabilities
    :return: Misclassification error (i.e. 1 - accuracy)
    """
    y = np.array(y)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    misclassification_error = np.mean(y != y_pred)
    return misclassification_error


def evaluate_models(
    y, model_MCMC, model_VB, model_MLE, model_glicko
):
    """
    Function to evaluate model performance on test data
    :param y: True labels
    :param model_MCMC: MCMC model (sample)
    :param model_VB: Variational Bayes model (variational)
    :param model_MLE: MLE model (optimize)
    :param glicko: Original Glicko model
    :return: Data frame of evaluation metrics
    """
    # Get model predictions on test sample
    y_pred_MCMC = get_posterior_predictions(
        model=model_MCMC,
        quantity="score_ppd"
    )
    y_pred_VB = get_posterior_predictions(
        model=model_VB,
        quantity="score_ppd"
    )
    y_pred_MLE = get_posterior_predictions(
        model=model_MLE,
        quantity="score_ppd"
    )
    y_pred_glicko = model_glicko
    # Compute binary cross entropy loss for each model
    bce_MCMC = get_binary_cross_entropy(y=y, y_pred=y_pred_MCMC)
    bce_VB = get_binary_cross_entropy(y=y, y_pred=y_pred_VB)
    bce_MLE = get_binary_cross_entropy(y=y, y_pred=y_pred_MLE)
    bce_glicko = get_binary_cross_entropy(y=y, y_pred=y_pred_glicko)
    # Compute missclassification error for each model
    mce_MCMC = get_misclassification_error(y=y, y_pred=y_pred_MCMC)
    mce_VB = get_misclassification_error(y=y, y_pred=y_pred_VB)
    mce_MLE = get_misclassification_error(y=y, y_pred=y_pred_MLE)
    mce_glicko = get_misclassification_error(y=y, y_pred=y_pred_glicko)
    # Combine results
    df = pd.DataFrame(
        data={
            " ":
            [
                "MCMC (HMC)",
                "MLE (L-BFGS)",
                "VB (Meanfield)",
                "Glicko2 (Original)"
            ],
            "binary cross entropy loss $$- \\dfrac{1}{n} \\sum y \times log(y_{pred}) \
            + (1-y) \times log(1 - y_{pred}) $$":
            [bce_MCMC, bce_MLE, bce_VB, bce_glicko],
            "misclassification error $$1 - \\dfrac{1}{n} \\sum \text{I}\\{y = y_{pred}\\} \
            $$": [mce_MCMC, mce_MLE, mce_VB, mce_glicko]
        }
    ).round(decimals=3).df.set_index(" ")
    df = df.style.set_caption("$$\\textbf{Testing Performance}$$")
    return df


def plot_ppc(score_ppc, observed_data, check, dim):
    """
    Function to plot and Bayesian p-values
    :param score_ppc: Synthetic scores
    :param observed_data: True data
    :param check: PP-Checks to apply
    :param dim: Marginalization dim (period or player)
    returns None
    """
    observed_data_ = dict()

    for (key, value) in observed_data.items():

        if 'test' not in key:
            observed_data_[key] = value

    observed_data = pd.DataFrame.from_dict(
        observed_data_
    )

    fig_size = 5

    fig, axes = plt.subplots(1, len(check))

    fig.set_size_inches(len(check) * fig_size, fig_size)

    bins = [33] * len(check)

    p_values = bayesian_p(score_ppc, observed_data, check, dim)

    for i in range(len(check)):
        main_title = 'T = {} (p-value : {:.2f})'.format(
            check[i], p_values[check[i]]
            )

        empirical_score = get_empirical_score(
            observed_data, check[i].lower(), dim
            )

        sim_dist = get_synthetic_score(
            score_ppc, observed_data, check[i].lower(), dim
            )

        axes[i].hist(sim_dist, bins=bins[i])

        axes[i].axvline(x=empirical_score, color='r')

        axes[i].title.set_text(main_title)

        axes[i].set_ylabel("{}".format('Number of wins'))

        fig.suptitle('Posterior Predictive Checks w.r.t. Period')

    plt.show()

    return None


def plot_trace(samples, gamma_1, gamma_2):
    """
    Function to plot trace figures
    :param samples: Latent r.v. samples
    :param gamma_1: Period index
    :param gamma_2: Player index
    returns None
    """
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
    """
    Function to plot elbo figure
    :param glicko_vi: Trained model
    returns None
    """
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
    """
    Function to plot ll figure
    :param glicko_map: Trained model
    returns None
    """
    for fname in glicko_map.runset._stdout_files:
        with open(fname, "r") as f:
            text = f.read()

        split = '\
            Iter      log prob        \
                ||dx||      ||grad||       alpha      alpha0  # evals  Notes '
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
    """
    Function to plot bce figure
    :param iteration: Iteration num
    :param bce: BCE val
    :param delta: Delta val
    """
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
