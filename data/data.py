from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import pickle
import os


def download_data(
    path="data/raw",
    force=True,
    quiet=True
):
    """
    Function to download project data from Kaggle
    Requires a Kaggle API token
    :param path: Path to store
    :param force: Whether to overwrite existing files
    :param quiet: Whether to run with feedback
    :return: None
    """
    # Authenticate to use the Kaggle Api
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    # Download the primary data file
    kaggle_api.competition_download_file(
        competition="ChessRatings2",
        file_name="followup_primary_training.zip",
        path="data/raw",
        force=force,
        quiet=quiet
    )
    # Download the secondary data file
    kaggle_api.competition_download_file(
        competition="ChessRatings2",
        file_name="followup_secondary_training.zip",
        path="data/raw",
        force=True,
        quiet=True
    )
    # Download the tertiary data file
    kaggle_api.competition_download_file(
        competition="ChessRatings2",
        file_name="followup_tertiary_training.zip",
        path="data/raw",
        force=True,
        quiet=True
    )


def process_data(
    player_num=103,
    period_length=12,
    perid_train=10,
    path_input="data/raw",
    path_output="data/processed"
):
    """
    Function to process our raw data
    :param player_num: Number of most frequent players to keep
    :param period_length: Number of months to combine into a signle period
    :param perid_train: Number of training period
    :param path_input: Path to fetch raw data from
    :param path_output: Path to save processed data to
    :return: None
    """
    # Import and concatenate our data sets
    data = pd.concat(
        [
            pd.read_csv(
                os.path.join(path_input, "followup_primary_training.zip"),
                header=0,
                names=["period", "white", "black", "score"],
                usecols=[1, 2, 3, 4]
            ),
            pd.read_csv(
                os.path.join(path_input, "followup_secondary_training.zip"),
                header=0,
                names=["period", "white", "black", "score"],
                usecols=[1, 2, 3, 4]
            ),
            pd.read_csv(
                os.path.join(path_input, "followup_tertiary_training.zip"),
                header=0,
                names=["period", "white", "black", "score"],
                usecols=[1, 2, 3, 4]
            )
        ]
    )
    # Remove games resulting in ties
    data = data[data["score"] != 0.5]
    # Collapse periods to every 'period_length' months
    data["period"] = ((data["period"] - 1) // period_length) + 1
    # Sort by period
    data = data.sort_values("period")
    # Find most frequent 'player_num' players
    players = pd.concat(
        [data["white"], data["black"]]
    ).value_counts()[:player_num].index.tolist()
    # Create a mapping most frequent 'player_num' players
    players_dict = dict(zip(players, list(range(1, player_num + 1))))
    # Filter data to most frequent 'player_num' players
    data = data[
        (data["white"].isin(players)) &
        (data["black"].isin(players))
    ]
    # Map most frequnet 'player_num' player IDs to indices
    # i.e. player with most games will have an ID of 1
    data["white"] = data["white"].map(players_dict)
    data["black"] = data["black"].map(players_dict)
    # Split data into training and testing sets
    # 'train' for model fitting and posterior predictive checks
    train = data[data["period"] <= perid_train]
    # 'test' for posterior predictive distribution
    test = data[data["period"] > perid_train]
    # Convert data to dictionary to use with Stan
    data_stan = {
        "n_game": train.shape[0],
        "n_period": perid_train,
        "n_player": player_num,
        "id_period": train["period"].tolist(),
        "id_white": train["white"].tolist(),
        "id_black": train["black"].tolist(),
        "score": train["score"].astype(int).tolist(),
        "n_game_test": test.shape[0],
        "id_white_test": test["white"].tolist(),
        "id_black_test": test["black"].tolist(),
        "score_test": test["score"].astype(int).tolist()
    }
    # Pickle and save the data dictionary
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)
    with open(os.path.join(path_output, "data.pkl"), "wb") as f:
        pickle.dump(data_stan, f)
