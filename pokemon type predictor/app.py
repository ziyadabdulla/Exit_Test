from pathlib import Path

import json
import math

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "G2 (1).csv"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"

NUMERIC_FEATURES = [
    "height_dm",
    "weight_hg",
    "base_experience",
    "hp",
    "attack",
    "defense",
]
INPUT_FEATURES = NUMERIC_FEATURES + ["type_2"]


def load_model():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not MODEL_INFO_PATH.is_file():
        raise FileNotFoundError(f"Model metadata not found: {MODEL_INFO_PATH}")

    with MODEL_INFO_PATH.open(encoding="utf-8") as model_info_file:
        model_info = json.load(model_info_file)

    dataset = pd.read_csv(DATASET_PATH)
    required_columns = set(INPUT_FEATURES + ["type_1"])
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")

    features = dataset[INPUT_FEATURES].copy()
    features["type_2"] = features["type_2"].fillna("unknown").replace("", "unknown")
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="raise"
    )
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].fillna(
        features[NUMERIC_FEATURES].median()
    )

    encoded_features = pd.get_dummies(features, columns=["type_2"])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(encoded_features)

    tuned_params = model_info.get("best_params", {})
    model = RandomForestClassifier(
        n_estimators=tuned_params.get("n_estimators", 100),
        max_depth=tuned_params.get("max_depth", 30),
        min_samples_split=tuned_params.get("min_samples_split", 5),
        random_state=42,
    )
    model.fit(scaled_features, dataset["type_1"])

    return model, scaler, encoded_features.columns.tolist(), sorted(
        features["type_2"].unique().tolist()
    ), model_info


model, scaler, model_columns, secondary_types, model_info = load_model()
app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html", model_info=model_info)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    form_values = request.form.to_dict()

    if request.method == "POST":
        try:
            row = {}
            for feature in NUMERIC_FEATURES:
                value = float(request.form[feature])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{feature.replace('_', ' ').title()} must be a non-negative number.")
                row[feature] = value

            secondary_type = request.form["type_2"]
            if secondary_type not in secondary_types:
                raise ValueError("Choose a secondary type from the list.")
            row["type_2"] = secondary_type

            encoded_row = pd.get_dummies(
                pd.DataFrame([row]), columns=["type_2"]
            ).reindex(columns=model_columns, fill_value=0)
            scaled_row = scaler.transform(encoded_row)
            prediction = model.predict(scaled_row)[0]
        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc) or "Enter valid values for every field."

    return render_template(
        "predict.html",
        prediction=prediction,
        error=error,
        form_values=form_values,
        secondary_types=secondary_types,
    )


if __name__ == "__main__":
    app.run()
