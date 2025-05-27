import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_train_test_data(data_df: pd.DataFrame, target_column: str, test_size: int = 0.25, random_state: int = 0) -> tuple:
    """Split training dataframe into train and test split."""
    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def preprocess_features(categorical_features: list, numeric_features: list | None = None) -> ColumnTransformer:
    """Apply preprocessing steps as a Machine Learning Pipeline."""
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    if numeric_features is None:
        preprocessor = ColumnTransformer(transformers=[("cat", categorical_transformer, categorical_features)])
    else:
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)]
        )
    return preprocessor


def train_model(x_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer, classifier: callable) -> callable:
    """Train the ml model."""
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return model.fit(x_train, y_train)
