"""Definições compartilhadas entre os notebooks: features, pipeline e o modelo em contagens."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMRegressor as _Model
    BACKEND = "lightgbm"
except ImportError:  # fallback sem LightGBM
    from sklearn.ensemble import RandomForestRegressor as _Model
    BACKEND = "sklearn"

LABEL = "cnt"
NUM_FEATURES = ["hour", "month", "holiday", "workingday", "temp", "atemp", "humidity", "windspeed"]
CAT_FEATURES = ["season", "weekday", "weather"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Colunas que NÃO podem virar feature: cnt = casual + registered; year só existe para a divisão temporal
LEAKY_COLUMNS = ["casual", "registered", "year", "date", "datetime"]


def rmse(y_true, y_pred) -> float:
    """RMSE (o scikit-learn 1.3, exigido pelo Deepchecks 0.19, não tem root_mean_squared_error)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Pré-processamento + regressor, treinado em log1p(cnt)."""
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)], remainder="drop")
    if BACKEND == "lightgbm":
        model = _Model(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                       random_state=random_state, verbose=-1)
    else:
        model = _Model(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state)
    return Pipeline([("preprocessor", pre), ("model", model)])


class CountModel:
    """Envolve a pipeline em log e devolve aluguéis por hora, com correção de viés log-normal.

    pred = exp(pred_log + 0.5 * sigma^2) - 1
    """

    def __init__(self, pipeline: Pipeline, sigma: float):
        self.pipeline = pipeline
        self.sigma = sigma

    def fit(self, X: pd.DataFrame, y):
        """Re-treina a pipeline em log1p(y). Mantém o sigma estimado na validação."""
        self.pipeline.fit(X[FEATURES], np.log1p(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_log = self.pipeline.predict(X[FEATURES])
        return np.clip(np.expm1(pred_log + 0.5 * self.sigma ** 2), 0, None)


class ScaledModel:
    """Recalibração: multiplica as previsões de um modelo base por um fator fixo.

    O fator é estimado em dados recentes (soma real / soma prevista), por exemplo nas últimas 4 semanas.
    """

    def __init__(self, base_model, factor: float):
        self.base_model = base_model
        self.factor = float(factor)

    def fit(self, X=None, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict(X) * self.factor
