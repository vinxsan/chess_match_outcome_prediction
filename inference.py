"""
Inference utilities for the chess match outcome project.

Public API:
    load_artifacts(models_dir=None)          -> dict with models, config and known openings
    build_features(data)                     -> DataFrame with engineered features
    predict_match_outcome(match_data, ...)   -> DataFrame with probabilities and matchmaking flag

The models are loaded once and cached, so repeated predictions do not re-read the files.
"""

import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# default models folder
MODELS_DIR = Path(__file__).resolve().parent / "models"

# number of moves assumed when estimating game duration
ASSUMED_MOVES = 40

# Raw columns needed to build the features. 'opening_moves' is also required,
# but it may be supplied under its raw dataset name 'opening_ply'
REQUIRED_COLUMNS = ["white_rating", "black_rating", "increment_code", "opening_eco", "rated"]


def categorize_time(minutes):
    """
    Categorize estimated game duration (in minutes) following the Lichess classification.

    Bullet: < 3, Blitz: 3 to < 10, Rapid: 10 to < 30, Classical: 30 or more.
    """
    if minutes < 3:
        return "Bullet"
    elif minutes < 10:
        return "Blitz"
    elif minutes < 30:
        return "Rapid"
    else:
        return "Classical"


def build_features(data):
    """
    Build the model features from raw pre-match columns.
    Takes a DataFrame with the columns white_rating, black_rating, increment_code,
    opening_eco, rated, and opening_ply (or opening_moves). Returns a new DataFrame
    with the same rows in the same order, plus the engineered columns: 
    base_time (minutes), increment_time_seconds, estimated_minutes, match_category
    and rating_diff.
    """
    df = data.copy()

    if df.empty:
        raise ValueError("No matches provided.")

    # the raw dataset calls this column 'opening_ply', the models were trained on 'opening_moves'
    if "opening_moves" not in df.columns and "opening_ply" in df.columns:
        df = df.rename(columns =  {"opening_ply": "opening_moves"})

    missing = sorted(set(REQUIRED_COLUMNS + ["opening_moves"]) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing} "
            "(note: 'opening_ply' is accepted in place of 'opening_moves')."
        )

    # split increment_code (e.g. "10+5") into base time (minutes) and increment (seconds)
    codes = df["increment_code"].astype(str)
    valid = codes.str.match(r"^\d+\+\d+$")
    if not valid.all():
        bad_values = df.loc[~valid, "increment_code"].tolist()
        raise ValueError(f"Invalid increment_code format (expected e.g. '10+0'): {bad_values}")

    split_time = codes.str.split("+", expand=True)
    df["base_time"] = split_time[0].astype(int)
    df["increment_time_seconds"] = split_time[1].astype(int)

    # estimated game duration in minutes: base time + increment for the assumed number of moves
    df["estimated_minutes"] = df["base_time"] + (df["increment_time_seconds"] * ASSUMED_MOVES / 60)
    df["match_category"] = df["estimated_minutes"].apply(categorize_time)

    df["rating_diff"] = df["white_rating"] - df["black_rating"]

    return df


def _get_known_openings(pipeline):
    """Return the set of opening_eco values seen by the OneHotEncoder during training."""
    cat_pipeline = pipeline.named_steps["preprocessing"].named_transformers_["cat"]
    encoder = cat_pipeline.named_steps["encoder"]
    position = list(encoder.feature_names_in_).index("opening_eco")
    return set(encoder.categories_[position])


def load_artifacts(models_dir=None):
    """Load models and config once; return everything needed for prediction"""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR

    required_files = ["three_classification_model.pkl", "binary_model.pkl", "config_params.json"]
    missing = [name for name in required_files if not (models_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {models_dir}: {missing}")

    with open(models_dir / "config_params.json") as f:
        config = json.load(f)

    three_class_model = joblib.load(models_dir / "three_classification_model.pkl")

    binary_model = joblib.load(models_dir / "binary_model.pkl")

    # the config class names must line up with the model's output columns
    if len(config["three_class_names"]) != len(three_class_model.classes_):
        raise ValueError("three_class_names in the config does not match the number of model classes")

    return {
        "three_class_model": three_class_model,
        "binary_model": binary_model,
        "class_names": config["three_class_names"],
        "binary_threshold": config["binary_threshold"],
        "known_openings": _get_known_openings(three_class_model),
    }


@lru_cache(maxsize=1)
def get_default_artifacts():
    """Load the artifacts from the default folder once and reuse them on later calls"""
    return load_artifacts()


def predict_match_outcome(match_data, artifacts=None):
    """
    Predict the outcome of one or more chess matches using only pre-match information

    match_data: a single dict, or a list of dicts, each with the keys
                'white_rating', 'black_rating', 'increment_code' (e.g. "10+0"),
                'opening_eco', 'opening_ply' (or 'opening_moves'), 'rated'.
    artifacts:  dictionary returned by load_artifacts(). If omitted, the models in the
                default models folder are loaded once and cached.

    Returns a DataFrame with one row per match, in input order: probabilities for each
    outcome (proba_black, proba_draw, proba_white), the surprise probability, the
    matchmaking flag and a data quality flag for unknown openings.
    """
    if artifacts is None:
        artifacts = get_default_artifacts()

    if isinstance(match_data, dict):
        match_data = [match_data]

    df = build_features(pd.DataFrame(match_data))

    # data quality check: flag opening_eco values unseen during training
    unknown_openings = ~df["opening_eco"].isin(artifacts["known_openings"])
    if unknown_openings.any():
        warnings.warn(
            f"{unknown_openings.sum()} match(es) have an opening_eco value not seen during training "
            "(treated as 'infrequent' by the model). This may indicate a data entry error "
            "or drift in opening popularity, worth monitoring over time.",
            UserWarning,
            stacklevel=2,
        )

    # three-class model: predict black / draw / white probabilities
    proba_3class = artifacts["three_class_model"].predict_proba(df)
    results = pd.DataFrame(
        proba_3class,
        columns=[f"proba_{name}" for name in artifacts["class_names"]],
        index=df.index,
    )

    # binary model: predict whether the favorite does NOT win (matchmaking use case)
    proba_binary = artifacts["binary_model"].predict_proba(df)[:, 1]
    results["surprise_probability"] = proba_binary
    results["flagged_as_potential_surprise"] = proba_binary >= artifacts["binary_threshold"]

    # carry the data quality flag into the output, for visibility/logging
    results["opening_eco_unknown"] = unknown_openings

    return results


if __name__ == "__main__":
    example = [
        {"white_rating": 1600, "black_rating": 1450, "increment_code": "10+0",
         "opening_eco": "B20", "opening_ply": 4, "rated": True},
        {"white_rating": 1500, "black_rating": 1500, "increment_code": "10+10",
         "opening_eco": "A00", "opening_ply": 10, "rated": False},
    ]
    print(predict_match_outcome(example))