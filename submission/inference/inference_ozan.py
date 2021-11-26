from cmdstanpy import CmdStanModel
import pickle
import os


def fit_model(
    stan_file,
    data,
    inference,
    path_output="model",
    save=False,
):
    """
    Function to conduct inference
    :param stan_file: Path to Stan model file
    :param data: Data dictionary to feed into Stan
    :param inference: inference type (MCMC, VB, or MLE)
    :param path_output: Path to save fitted model to
    :param save: Whether to save fitted model
    :return: Stan model
    """
    # Compile the Stan model
    glicko = CmdStanModel(stan_file=stan_file)
    # Run the model using desired inference method
    if inference == "MCMC":
        # HMC with NUTS
        model = glicko.sample(
            data=data,
            seed=147,
            chains=4,
            parallel_chains=4,
            adapt_delta=0.8,
            refresh=500,
            iter_warmup=1000,
            iter_sampling=1000
        )
    elif inference == "VB":
        # Meanfield Variational Inference
        model = glicko.variational(
            data=data,
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
    elif inference == "MLE":
        # L-BFGS Optimization
        model = glicko.optimize(
            data=data,
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
    else:
        raise ValueError("inference must be MCMC, VB, or MLE")
    # Pickle and save the model
    if save:
        if not os.path.exists(path_output):
            os.makedirs(path_output, exist_ok=True)
        with open(os.path.join(path_output, inference + ".pkl"), "wb") as f:
            pickle.dump(model, f)
    return(model)
