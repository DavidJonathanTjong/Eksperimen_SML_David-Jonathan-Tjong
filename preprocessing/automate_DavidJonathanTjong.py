import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
ORDINAL_FEATURES = ["education"]
EDUCATION_ORDER = [["unknown", "primary", "secondary", "tertiary"]]
NOMINAL_FEATURES = ["job", "marital", "default", "housing", "loan", "contact", "month", "poutcome"]


def load_raw_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep=";")
    print(f"Data loaded successfully! Shape: {df.shape}")
    return df


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("ordinal", OrdinalEncoder(categories=EDUCATION_ORDER), ORDINAL_FEATURES),
        ("nominal", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), NOMINAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor)])


def preprocess_data(df):
    df_clean = df.drop_duplicates().drop(columns=["duration"])
    df_clean["y"] = df_clean["y"].map({"yes": 1, "no": 0})

    X = df_clean.drop(columns=["y"])
    y = df_clean["y"]

    pipeline = build_pipeline()
    X_processed = pipeline.fit_transform(X)

    onehot_cols = pipeline.named_steps["preprocessor"].named_transformers_["nominal"] \
        .get_feature_names_out(NOMINAL_FEATURES)
    columns = NUMERIC_FEATURES + ORDINAL_FEATURES + list(onehot_cols)

    df_processed = pd.DataFrame(X_processed, columns=columns, index=X.index)
    df_processed["y"] = y.values

    print(f"Preprocessing completed! Shape: {df_processed.shape}")
    return df_processed


def save_preprocessed_data(df, output_path):
    """Save preprocessed data to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def main():
    base_dir = Path(__file__).parent.parent
    raw_path = base_dir / "bank-full.csv"
    output_path = base_dir / "preprocessing" / "bank-full_preprocessing.csv"

    df = load_raw_data(raw_path)
    df_processed = preprocess_data(df)
    save_preprocessed_data(df_processed, output_path)


if __name__ == "__main__":
    main()
