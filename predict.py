from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent
BODY_MODEL_PATH = ROOT / "models" / "position_model.joblib"
STATS_MODEL_PATH = ROOT / "models" / "position_model_with_stats.joblib"
PLAYERS_WITH_STATS = ROOT / "data" / "processed" / "players_with_stats.csv"


def add_features(df):
    df = df.copy()
    df["ape_index_cm"] = df["wingspan_cm"] - df["height_cm"]
    df["reach_minus_height_cm"] = df["reach_cm"] - df["height_cm"]
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    return df


def load_model(path=BODY_MODEL_PATH):
    return joblib.load(path)


def _probabilities(bundle, rows):
    model = bundle["model"]
    probs = model.predict_proba(rows[bundle["features"]])[0]
    return pd.Series(probs, index=model.classes_).sort_values(ascending=False)


def predict_position(height_cm, weight_kg, wingspan_cm=None, reach_cm=None, body_fat_pct=None,
                     hand_length_cm=None, hand_width_cm=None, bundle=None):
    row = pd.DataFrame([{
        "height_cm": height_cm, "weight_kg": weight_kg, "wingspan_cm": wingspan_cm, "reach_cm": reach_cm,
        "body_fat_pct": body_fat_pct, "hand_length_cm": hand_length_cm, "hand_width_cm": hand_width_cm,
    }], dtype=float)
    return _probabilities(bundle or load_model(), add_features(row))


def predict_known_player(name, bundle=None):
    players = pd.read_csv(PLAYERS_WITH_STATS)
    row = players[players["player"].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"{name!r} not found in {PLAYERS_WITH_STATS.name}")
    return _probabilities(bundle or load_model(STATS_MODEL_PATH), row)
