from cmdstanpy.stanfit import CmdStanMCMC, CmdStanVB, CmdStanMLE
import pandas as pd
import numpy as np


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
    return(y_pred)


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
    return(binomial_deviance)


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
    return(misclassification_error)


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
    # Compute binary cross entropy loss for each model
    bce_MCMC = get_binary_cross_entropy(y=y, y_pred=y_pred_MCMC)
    bce_VB = get_binary_cross_entropy(y=y, y_pred=y_pred_VB)
    bce_MLE = get_binary_cross_entropy(y=y, y_pred=y_pred_MLE)
    # Compute missclassification error for each model
    mce_MCMC = get_misclassification_error(y=y, y_pred=y_pred_MCMC)
    mce_VB = get_misclassification_error(y=y, y_pred=y_pred_VB)
    mce_MLE = get_misclassification_error(y=y, y_pred=y_pred_MLE)
    # Combine results
    df = pd.DataFrame(
        data={
            " ": ["MCMC (HMC)", "MLE (L-BFGS)", "VB (Meanfield)"],
            "binary cross entropy loss $$- \\dfrac{1}{n} \\sum y \times log(y_{pred}) \
            + (1-y) \times log(1 - y_{pred}) $$": [bce_MCMC, bce_MLE, bce_VB],
            "misclassification error $$1 - \\dfrac{1}{n} \\sum \text{I}\\{y = y_{pred}\\} \
            $$": [mce_MCMC, mce_MLE, mce_VB]
        }
    ).round(decimals=3).set_index(" ")
    return(df)
