
# ============================================================
# ODISHA RICE YIELD ANALYTICS DASHBOARD
# Shared Utilities
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
LIME_DIR = ARTIFACT_DIR / "lime"
PHASE4_DIR = ARTIFACT_DIR / "phase4"

DATA_PATH = PROJECT_ROOT / "clean_rice.csv"
MODEL_PATH = ARTIFACT_DIR / "final_model_rf.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


# ============================================================
# 2. FEATURE CONFIGURATION
# ============================================================

NUMERICAL_FEATURES = [
    "area",
    "year",
    "avg_temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "soil_ph",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
    "irrigation_pct"
]

CATEGORICAL_FEATURES = [
    "district",
    "season"
]

FEATURE_COLUMNS = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)

TARGET_COLUMN = "yield_kg_ha"

TRAINING_SEASONS = [
    "Autumn",
    "Summer",
    "Winter"
]

ALL_SEASONS = [
    "Summer",
    "Winter",
    "Autumn",
    "Kharif"
]


# ============================================================
# 3. FILE VALIDATION
# ============================================================

def check_file_exists(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return path


# ============================================================
# 4. LOAD DATASET
# ============================================================

@st.cache_data(show_spinner="Loading rice yield dataset...")
def load_dataset():

    check_file_exists(DATA_PATH)

    data = pd.read_csv(DATA_PATH)

    required_columns = (
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    return data


# ============================================================
# 5. LOAD MODEL METADATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_metadata():

    check_file_exists(METADATA_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return metadata


# ============================================================
# 6. LOAD SAVED MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading trained ML model...")
def load_model():

    check_file_exists(MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Saved model does not have a predict method."
        )

    return model


# ============================================================
# 7. LOAD CSV ARTIFACTS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv_artifact(relative_path):

    path = ARTIFACT_DIR / relative_path

    check_file_exists(path)

    return pd.read_csv(path)


# ============================================================
# 8. LOAD JSON ARTIFACTS
# ============================================================

@st.cache_data(show_spinner=False)
def load_json_artifact(relative_path):

    path = ARTIFACT_DIR / relative_path

    check_file_exists(path)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# 9. LOAD IMAGE ARTIFACTS
# ============================================================

def get_figure_path(filename):

    path = FIGURE_DIR / filename

    check_file_exists(path)

    return str(path)


# ============================================================
# 10. LOAD LIME HTML FILE
# ============================================================

def get_lime_html_path(filename):

    path = LIME_DIR / "html" / filename

    check_file_exists(path)

    return str(path)


# ============================================================
# 11. GENERATE MODEL PREDICTIONS
# ============================================================

def predict_yield(input_data):

    model = load_model()

    if isinstance(input_data, dict):

        input_data = pd.DataFrame(
            [input_data]
        )

    elif not isinstance(input_data, pd.DataFrame):

        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in input_data.columns
    ]

    extra_features = [
        col for col in input_data.columns
        if col not in FEATURE_COLUMNS
    ]

    if missing_features or extra_features:

        raise ValueError(
            f"Invalid prediction features. "
            f"Missing: {missing_features}. "
            f"Extra: {extra_features}."
        )

    # Preserve the original model feature order
    input_data = input_data[FEATURE_COLUMNS].copy()

    if input_data.isnull().any().any():

        raise ValueError(
            "Prediction inputs cannot contain missing values."
        )

    predictions = model.predict(input_data)

    return np.asarray(predictions)


# ============================================================
# 12. VALIDATE PREDICTION INPUTS
# ============================================================

def validate_prediction_inputs(input_data):

    if isinstance(input_data, dict):

        input_data = pd.DataFrame(
            [input_data]
        )

    if list(input_data.columns) != FEATURE_COLUMNS:

        return False

    if input_data.isnull().any().any():

        return False

    return True


# ============================================================
# 13. FORMAT NUMBERS
# ============================================================

def format_yield(value):

    return f"{value:,.2f} kg/ha"


def format_area(value):

    return f"{value:,.2f} ha"


def format_production(value):

    return f"{value:,.2f} tonnes"


# ============================================================
# 14. Kharif WARNING
# ============================================================

def is_unseen_season(season):

    return season not in TRAINING_SEASONS


# ============================================================
# 15. DASHBOARD CONSTANTS
# ============================================================

DASHBOARD_TITLE = "Odisha Rice Yield Analytics"

DASHBOARD_SUBTITLE = (
    "District-Level Prediction, Agricultural Trends "
    "and Explainable Machine Learning"
)

BASELINE_METRICS = {
    "rmse": 541.7697395333312,
    "mae": 435.91360486142503,
    "r2": 0.4165031866242306
}
