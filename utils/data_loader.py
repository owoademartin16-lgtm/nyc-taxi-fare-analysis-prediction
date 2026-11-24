"""Cached data and model access for the taxi dashboard."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "taxi_model.joblib"
PREPROCESSOR_PATH = MODELS / "taxi_preprocessor.joblib"
FALLBACK_PATH = MODELS / "taxi_fallback.joblib"


def _install_sklearn_compat() -> None:
    """Restore private names removed after scikit-learn 1.5.

    The saved preprocessor and fallback were fitted with scikit-learn 1.5.2.
    Newer versions no longer expose ``_RemainderColsList`` and the fallback
    pickle refers to the old ``_loss`` module name.
    """
    try:
        import sklearn.compose._column_transformer as column_transformer
    except Exception:
        column_transformer = None
    if column_transformer is not None and not hasattr(
        column_transformer, "_RemainderColsList"
    ):
        class _RemainderColsList(list):
            def __init__(self, iterable=(), *, future_dtype=None):
                self.future_dtype = future_dtype
                super().__init__(iterable)

        column_transformer._RemainderColsList = _RemainderColsList

    try:
        import sklearn._loss.loss as loss

        sys.modules.setdefault("_loss", loss)
    except Exception:
        pass


def _repair_preprocessor(preprocessor):
    """Restore passthrough columns when an older ColumnTransformer is unpickled."""
    try:
        if getattr(preprocessor, "remainder", None) != "passthrough":
            return preprocessor
        transformers = getattr(preprocessor, "transformers_", None)
        feature_names = getattr(preprocessor, "feature_names_in_", None)
        if not transformers or feature_names is None:
            return preprocessor

        explicit = set()
        for _, _, columns in transformers:
            explicit.update(columns)
        missing = [name for name in feature_names if name not in explicit]
        if not missing:
            return preprocessor

        name, transformer, columns = transformers[-1]
        if len(columns):
            return preprocessor

        transformers[-1] = (name, transformer, missing)
        preprocessor._remainder = (name, transformer, missing)
        preprocessor._columns = [cols for _, _, cols in transformers]
    except Exception:
        pass
    return preprocessor



@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    return json.loads((DATA / "summary.json").read_text())


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    return json.loads((DATA / "model_metrics.json").read_text())


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    """Load one of the small aggregate CSVs from data/."""
    return pd.read_csv(DATA / name)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_parquet(DATA / "taxi_sample.parquet")


@st.cache_data(show_spinner=False)
def load_model_sample() -> pd.DataFrame:
    return pd.read_parquet(DATA / "model_test_sample.parquet")


@st.cache_data(show_spinner=False)
def load_zones() -> pd.DataFrame:
    return pd.read_csv(DATA / "zones.csv")


@st.cache_data(show_spinner=False)
def load_zone_geojson() -> dict:
    return json.loads((DATA / "zone_map.geojson").read_text())


@st.cache_data(show_spinner=False)
def load_zone_lookup() -> pd.DataFrame:
    return pd.read_csv(DATA / "taxi_zone_lookup.csv", encoding="utf-8-sig")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the saved model and preprocessor, with a portable fallback.

    LightGBM is the deployed model. On a local macOS Python without libomp the
    import can fail; in that case the dashboard uses a scikit-learn model trained
    on the same features so the estimator remains usable. Deployment uses the
    original LightGBM artifact.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
    _install_sklearn_compat()
    preprocessor = _repair_preprocessor(joblib.load(PREPROCESSOR_PATH))
    try:
        import lightgbm  # noqa: F401

        model = joblib.load(MODEL_PATH)
        return model, preprocessor, "LightGBM", False
    except Exception:
        if not FALLBACK_PATH.exists():
            raise
        model = joblib.load(FALLBACK_PATH)
        return model, preprocessor, "HistGradientBoosting", True


def dataset_info() -> dict:
    summary = load_summary()
    metrics = load_model_metrics()
    return {
        "trips": summary["total_trips"],
        "revenue": summary["revenue"],
        "r2": metrics["metrics"]["r2"],
        "mae": metrics["metrics"]["mae"],
        "test_rows": metrics["split"]["test"],
    }
