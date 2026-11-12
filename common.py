"""Fungsi bersama: memuat dataset dan membuat pipeline ANN."""
from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "IRIS.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
NAMA_KELAS = {0: "setosa", 1: "versicolor", 2: "virginica"}
NAMA_KE_ID = {v: k for k, v in NAMA_KELAS.items()}


def load_dataset():
    """Baca IRIS.csv. Jika kolom species_id belum ada, dibuat dari kolom
    species (Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2)."""
    df = pd.read_csv(DATASET_PATH)
    df["species"] = df["species"].str.strip().str.replace("Iris-", "", regex=False)
    if "species_id" not in df.columns:
        df["species_id"] = df["species"].map(NAMA_KE_ID)
    return df


def build_model(hidden=(10,), activation="relu"):
    """Pipeline: StandardScaler + MLPClassifier (Adam)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ann", MLPClassifier(
            hidden_layer_sizes=hidden, activation=activation,
            solver="adam", learning_rate_init=0.01,
            max_iter=1500, early_stopping=True,
            n_iter_no_change=30, random_state=42,
        )),
    ])
