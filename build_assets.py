"""Build the NYC taxi dashboard assets from the public TLC January 2025 file.

This script is the reproducible data layer for the dashboard. It downloads the
official Yellow Taxi parquet and zone lookup, applies the cleaning recipe from
the case study, and writes the compact aggregates, samples and model
diagnostics that the Streamlit pages read.

Run once from this folder:

    python3 build_assets.py

The dashboard itself never trains or downloads anything at runtime.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DATA = ROOT / "data"
MODELS = ROOT / "models"

TRIPS_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2025-01.parquet"
)
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

FEATURES = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_day_type",
]

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

PAYMENT_LABELS = {
    0: "Unknown",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Voided",
    6: "Unknown",
}

RATECODE_LABELS = {
    1.0: "Standard meter",
    2.0: "JFK flat fare",
    3.0: "Newark",
    4.0: "Nassau / Westchester",
    5.0: "Negotiated",
    6.0: "Group ride",
    99.0: "Unknown",
}

FARE_BINS = [0, 15, 25, 40, 70, 1_000]
FARE_LABELS = ["<$15", "$15–25", "$25–40", "$40–70", "$70+"]
DIST_BINS = [0, 1, 2, 3, 5, 10, 20, 101]
DIST_LABELS = ["<1 mi", "1–2 mi", "2–3 mi", "3–5 mi", "5–10 mi", "10–20 mi", "20+ mi"]


# ------------------------------------------------------------------ helpers ---
def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Using cached {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def hour_label(hour: int) -> str:
    if hour == 0:
        return "12a"
    if hour < 12:
        return f"{hour}a"
    if hour == 12:
        return "12p"
    return f"{hour - 12}p"


def shift_for(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 22:
        return "Evening"
    return "Overnight"


def clean_eda(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning recipe used for the exploratory story."""
    df = raw[
        (raw["tpep_pickup_datetime"] >= "2025-01-01")
        & (raw["tpep_pickup_datetime"] < "2025-02-01")
    ].copy()

    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1})
    df["passenger_count"] = df["passenger_count"].fillna(
        df["passenger_count"].median()
    )
    df["RatecodeID"] = df["RatecodeID"].fillna(df["RatecodeID"].mode()[0])
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna(
        df["store_and_fwd_flag"].mode()[0]
    )
    df["congestion_surcharge"] = df["congestion_surcharge"].fillna(
        df["congestion_surcharge"].median()
    )
    df["Airport_fee"] = df["Airport_fee"].fillna(df["Airport_fee"].median())

    df = df[(df["fare_amount"] >= 0) & (df["total_amount"] >= 0)].copy()

    dt = df["tpep_pickup_datetime"]
    df["pickup_year"] = dt.dt.year
    df["pickup_month"] = dt.dt.month
    df["pickup_day"] = dt.dt.day
    df["pickup_day_of_week"] = dt.dt.day_name()
    df["pickup_hour"] = dt.dt.hour
    df["pickup_day_type"] = np.where(dt.dt.dayofweek >= 5, "Weekend", "Weekday")
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df["fare_per_mile"] = np.where(
        df["trip_distance"] > 0, df["fare_amount"] / df["trip_distance"], np.nan
    )
    return df


def model_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Physical limits used by the case-study model."""
    return df[
        (df["total_amount"] > 0)
        & (df["total_amount"] <= 300)
        & (df["trip_distance"] > 0)
        & (df["trip_distance"] <= 100)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    ].copy()


# --------------------------------------------------------------- eda assets ---
def build_eda_assets(df: pd.DataFrame) -> dict:
    n = len(df)
    summary = {
        "total_trips": int(n),
        "revenue": float(df["total_amount"].sum()),
        "avg_total": float(df["total_amount"].mean()),
        "median_total": float(df["total_amount"].median()),
        "avg_fare": float(df["fare_amount"].mean()),
        "median_fare": float(df["fare_amount"].median()),
        "avg_distance": float(df["trip_distance"].mean()),
        "median_distance": float(df["trip_distance"].median()),
        "avg_passengers": float(df["passenger_count"].mean()),
        "solo_share": float((df["passenger_count"] == 1).mean()),
        "card_share": float((df["payment_type"] == 1).mean()),
        "card_tip_rate": float(
            df.loc[df["payment_type"] == 1, "tip_amount"].sum()
            / df.loc[df["payment_type"] == 1, "total_amount"].sum()
        ),
        "weekday_trips": int((df["pickup_day_type"] == "Weekday").sum()),
        "weekend_trips": int((df["pickup_day_type"] == "Weekend").sum()),
        "weekday_share": float((df["pickup_day_type"] == "Weekday").mean()),
        "weekday_revenue": float(
            df.loc[df["pickup_day_type"] == "Weekday", "total_amount"].sum()
        ),
        "weekend_revenue": float(
            df.loc[df["pickup_day_type"] == "Weekend", "total_amount"].sum()
        ),
    }

    hourly = (
        df.groupby("pickup_hour")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            median_total=("total_amount", "median"),
            avg_fare=("fare_amount", "mean"),
            median_fare=("fare_amount", "median"),
            avg_distance=("trip_distance", "mean"),
            avg_tip=("tip_amount", "mean"),
        )
        .reindex(range(24))
        .reset_index()
    )
    hourly["hour_label"] = hourly["pickup_hour"].map(hour_label)

    split = (
        df.groupby(["pickup_hour", "pickup_day_type"])
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_fare=("fare_amount", "mean"),
        )
        .reset_index()
        .pivot(index="pickup_hour", columns="pickup_day_type")
    )
    split.columns = [f"{a}_{b.lower()}" for a, b in split.columns]
    hourly = hourly.merge(split.reset_index(), on="pickup_hour", how="left")
    hourly.to_csv(DATA / "hourly.csv", index=False)

    peak_hour = int(hourly.loc[hourly["trips"].idxmax(), "pickup_hour"])
    summary["peak_hour"] = peak_hour
    summary["peak_hour_trips"] = int(hourly["trips"].max())
    summary["peak_hour_revenue"] = float(
        hourly.loc[hourly["pickup_hour"] == peak_hour, "revenue"].iloc[0]
    )
    fare_hour = int(hourly.loc[hourly["avg_fare"].idxmax(), "pickup_hour"])
    summary["highest_avg_fare_hour"] = fare_hour
    summary["highest_avg_fare"] = float(hourly["avg_fare"].max())

    daily = (
        df.groupby("pickup_day")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
            avg_tip=("tip_amount", "mean"),
        )
        .reindex(range(1, 32))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(
        {"year": 2025, "month": 1, "day": daily["pickup_day"]}
    )
    daily["weekday"] = daily["date"].dt.day_name()
    daily["day_type"] = np.where(
        daily["date"].dt.dayofweek >= 5, "Weekend", "Weekday"
    )
    daily["trips"] = daily["trips"].fillna(0).astype(int)
    daily["revenue"] = daily["revenue"].fillna(0)
    daily["avg_total"] = daily["avg_total"].fillna(0)
    daily["avg_fare"] = daily["avg_fare"].fillna(0)
    daily["avg_distance"] = daily["avg_distance"].fillna(0)
    daily["avg_tip"] = daily["avg_tip"].fillna(0)
    daily.to_csv(DATA / "daily.csv", index=False)

    peak_day = int(daily.loc[daily["trips"].idxmax(), "pickup_day"])
    summary["peak_day"] = peak_day
    summary["peak_day_trips"] = int(daily["trips"].max())
    summary["peak_day_revenue"] = float(
        daily.loc[daily["pickup_day"] == peak_day, "revenue"].iloc[0]
    )

    dow = (
        df.groupby("pickup_day_of_week")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_fare=("fare_amount", "mean"),
            median_fare=("fare_amount", "median"),
            avg_distance=("trip_distance", "mean"),
            avg_tip=("tip_amount", "mean"),
        )
        .reindex(DAY_ORDER)
        .reset_index()
    )
    dow["day_type"] = np.where(
        dow["pickup_day_of_week"].isin(["Saturday", "Sunday"]),
        "Weekend",
        "Weekday",
    )
    dow.to_csv(DATA / "dow.csv", index=False)
    summary["highest_avg_fare_day"] = str(dow.loc[dow["avg_fare"].idxmax(), "pickup_day_of_week"])
    summary["highest_avg_fare_day_value"] = float(dow["avg_fare"].max())

    heat = (
        df.groupby(["pickup_day_of_week", "pickup_hour"])
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
        )
        .reset_index()
    )
    heat["pickup_day_of_week"] = pd.Categorical(
        heat["pickup_day_of_week"], categories=DAY_ORDER, ordered=True
    )
    heat = heat.sort_values(["pickup_day_of_week", "pickup_hour"])
    heat.to_csv(DATA / "heatmap.csv", index=False)

    daytype = (
        df.groupby("pickup_day_type")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
            avg_tip=("tip_amount", "mean"),
        )
        .reindex(["Weekday", "Weekend"])
        .reset_index()
    )
    daytype["share"] = daytype["trips"] / daytype["trips"].sum()
    daytype.to_csv(DATA / "daytype.csv", index=False)

    shifts = df.copy()
    shifts["shift"] = shifts["pickup_hour"].map(shift_for)
    shift_table = (
        shifts.groupby("shift")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
        )
        .reindex(["Overnight", "Morning", "Afternoon", "Evening"])
        .reset_index()
    )
    shift_table["share"] = shift_table["trips"] / shift_table["trips"].sum()
    shift_table.to_csv(DATA / "shifts.csv", index=False)

    payment = (
        df.assign(payment_label=df["payment_type"].map(PAYMENT_LABELS).fillna("Unknown"))
        .groupby("payment_label")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_tip=("tip_amount", "mean"),
            tip_total=("tip_amount", "sum"),
        )
        .reset_index()
        .sort_values("trips", ascending=False)
    )
    payment["share"] = payment["trips"] / payment["trips"].sum()
    payment["tip_rate"] = payment["tip_total"] / payment["revenue"].replace(0, np.nan)
    payment.to_csv(DATA / "payment.csv", index=False)

    ratecode = (
        df.assign(rate_label=df["RatecodeID"].map(RATECODE_LABELS).fillna("Unknown"))
        .groupby(["RatecodeID", "rate_label"])
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
            avg_tip=("tip_amount", "mean"),
        )
        .reset_index()
        .sort_values("trips", ascending=False)
    )
    ratecode["share"] = ratecode["trips"] / ratecode["trips"].sum()
    ratecode["revenue_share"] = ratecode["revenue"] / ratecode["revenue"].sum()
    ratecode.to_csv(DATA / "ratecode.csv", index=False)
    airport = ratecode[ratecode["RatecodeID"].isin([2.0, 3.0])]
    summary["airport_trips"] = int(airport["trips"].sum())
    summary["airport_revenue"] = float(airport["revenue"].sum())
    summary["airport_share"] = float(airport["trips"].sum() / n)
    summary["airport_revenue_share"] = float(airport["revenue"].sum() / summary["revenue"])

    passengers = (
        df.groupby("passenger_count")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
        )
        .reset_index()
        .sort_values("passenger_count")
    )
    passengers["share"] = passengers["trips"] / passengers["trips"].sum()
    passengers.to_csv(DATA / "passengers.csv", index=False)

    zones = pd.read_csv(DATA / "taxi_zone_lookup.csv", encoding="utf-8-sig")
    centroids = pd.read_csv(DATA / "zone_centroids.csv")[["LocationID", "lon", "lat"]]
    zones = zones.merge(centroids, on="LocationID", how="left")

    pickup = (
        df.groupby("PULocationID")
        .agg(
            pickup_trips=("total_amount", "size"),
            pickup_revenue=("total_amount", "sum"),
            pickup_avg_total=("total_amount", "mean"),
            pickup_avg_distance=("trip_distance", "mean"),
        )
        .reset_index()
        .rename(columns={"PULocationID": "LocationID"})
    )
    dropoff = (
        df.groupby("DOLocationID")
        .agg(
            dropoff_trips=("total_amount", "size"),
            dropoff_revenue=("total_amount", "sum"),
            dropoff_avg_total=("total_amount", "mean"),
            dropoff_avg_distance=("trip_distance", "mean"),
        )
        .reset_index()
        .rename(columns={"DOLocationID": "LocationID"})
    )
    zones = zones.merge(pickup, on="LocationID", how="left").merge(
        dropoff, on="LocationID", how="left"
    )
    for col in [
        "pickup_trips",
        "pickup_revenue",
        "pickup_avg_total",
        "pickup_avg_distance",
        "dropoff_trips",
        "dropoff_revenue",
        "dropoff_avg_total",
        "dropoff_avg_distance",
    ]:
        zones[col] = zones[col].fillna(0)
    zones["pickup_trips"] = zones["pickup_trips"].astype(int)
    zones["dropoff_trips"] = zones["dropoff_trips"].astype(int)
    zones["Zone"] = zones["Zone"].fillna("Unknown")
    zones["Borough"] = zones["Borough"].fillna("Unknown")
    zones["service_zone"] = zones["service_zone"].fillna("Unknown")
    zones.to_csv(DATA / "zones.csv", index=False)

    borough = (
        df.merge(
            zones[["LocationID", "Borough"]],
            left_on="PULocationID",
            right_on="LocationID",
            how="left",
        )
        .groupby("Borough")
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
        )
        .reset_index()
        .sort_values("trips", ascending=False)
    )
    borough["share"] = borough["trips"] / borough["trips"].sum()
    borough.to_csv(DATA / "boroughs.csv", index=False)

    pairs = (
        df.groupby(["PULocationID", "DOLocationID"])
        .agg(
            trips=("total_amount", "size"),
            revenue=("total_amount", "sum"),
            avg_total=("total_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
        )
        .reset_index()
        .sort_values("trips", ascending=False)
        .head(30)
    )
    pairs = pairs.merge(
        zones[["LocationID", "Zone", "Borough"]].rename(
            columns={
                "LocationID": "PULocationID",
                "Zone": "PickupZone",
                "Borough": "PickupBorough",
            }
        ),
        on="PULocationID",
        how="left",
    ).merge(
        zones[["LocationID", "Zone", "Borough"]].rename(
            columns={
                "LocationID": "DOLocationID",
                "Zone": "DropoffZone",
                "Borough": "DropoffBorough",
            }
        ),
        on="DOLocationID",
        how="left",
    )
    pairs.to_csv(DATA / "top_pairs.csv", index=False)

    summary["weekday_share"] = float(summary["weekday_share"])
    summary["solo_share"] = float(summary["solo_share"])
    summary["card_tip_rate"] = float(summary["card_tip_rate"])
    (DATA / "summary_eda.json").write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------- model assets ---
def _feature_label(name: str) -> str:
    mapping = {
        "remainder__VendorID": "Vendor",
        "remainder__passenger_count": "Passengers",
        "remainder__trip_distance": "Trip distance",
        "remainder__RatecodeID": "Rate code",
        "remainder__store_and_fwd_flag": "Store & forward",
        "remainder__PULocationID": "Pickup zone",
        "remainder__DOLocationID": "Drop-off zone",
        "remainder__payment_type": "Payment type",
        "remainder__pickup_year": "Pickup year",
        "remainder__pickup_month": "Pickup month",
        "remainder__pickup_day": "Day of month",
        "remainder__pickup_hour": "Pickup hour",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("cat__pickup_day_of_week_"):
        return "Day of week"
    if name.startswith("cat__pickup_day_type_"):
        return "Day type"
    return name.replace("remainder__", "").replace("cat__", "")


def _band_table(frame: pd.DataFrame, band_col: str, labels: list[str]) -> list[dict]:
    table = (
        frame.groupby(band_col, observed=True)
        .agg(
            trips=("actual", "size"),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
            p10=("residual", lambda s: float(s.quantile(0.10))),
            p90=("residual", lambda s: float(s.quantile(0.90))),
        )
        .reindex(labels)
        .reset_index()
    )
    total = table["trips"].sum()
    table["share"] = table["trips"] / total
    return [
        {
            "band": str(row[band_col]),
            "trips": int(row["trips"]),
            "share": float(row["share"]),
            "mae": float(row["mae"]),
            "bias": float(row["bias"]),
            "p10": float(row["p10"]),
            "p90": float(row["p90"]),
        }
        for _, row in table.iterrows()
    ]


def build_model_assets(df: pd.DataFrame) -> dict | None:
    """Reproduce the saved LightGBM holdout and write diagnostics."""
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"Skipping model diagnostics: LightGBM unavailable ({exc})")
        return None

    model = joblib.load(MODELS / "taxi_model.joblib")
    preprocessor = joblib.load(MODELS / "taxi_preprocessor.joblib")

    X = df[FEATURES]
    y = df["total_amount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    pred_train = model.predict(X_train_t)
    pred = model.predict(X_test_t)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "train_r2": float(r2_score(y_train, pred_train)),
        "gap": float(r2_score(y_train, pred_train) - r2_score(y_test, pred)),
    }

    result = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": pred,
            "residual": y_test.to_numpy() - pred,
            "abs_error": np.abs(y_test.to_numpy() - pred),
            "trip_distance": X_test["trip_distance"].to_numpy(),
            "pickup_hour": X_test["pickup_hour"].to_numpy(),
            "pickup_day_type": X_test["pickup_day_type"].to_numpy(),
            "passenger_count": X_test["passenger_count"].to_numpy(),
        }
    )
    result["fare_band"] = pd.cut(
        result["actual"], bins=FARE_BINS, labels=FARE_LABELS, right=False
    )
    result["distance_band"] = pd.cut(
        result["trip_distance"], bins=DIST_BINS, labels=DIST_LABELS, right=False
    )

    rng = np.random.default_rng(42)
    idx = rng.choice(len(result), size=min(25_000, len(result)), replace=False)
    result.iloc[idx].to_parquet(DATA / "model_test_sample.parquet", index=False)

    try:
        importance = model.booster_.feature_importance(importance_type="split")
        names = model.feature_name_
    except Exception:
        importance = model.feature_importances_
        names = preprocessor.get_feature_names_out()

    fi = pd.DataFrame({"name": names, "gain": importance})
    fi["feature"] = fi["name"].map(_feature_label)
    fi = (
        fi.groupby("feature", as_index=False)["gain"]
        .sum()
        .sort_values("gain", ascending=False)
    )
    fi["share"] = fi["gain"] / fi["gain"].sum()
    feature_importance = [
        {
            "feature": row["feature"],
            "importance": float(row["gain"]),
            "share": float(row["share"]),
        }
        for _, row in fi.iterrows()
    ]

    residuals = result["residual"]
    payload = {
        "metrics": metrics,
        "comparison": [
            {
                "model": "Linear Regression",
                "mae": 4.13,
                "rmse": 7.56,
                "r2": 0.8589,
                "status": "Benchmark",
            },
            {
                "model": "HistGradientBoosting",
                "mae": 2.62,
                "rmse": 4.63,
                "r2": 0.9469,
                "status": "Challenger",
            },
            {
                "model": "LightGBM",
                "mae": round(metrics["mae"], 2),
                "rmse": round(metrics["rmse"], 2),
                "r2": round(metrics["r2"], 4),
                "status": "Deployed",
            },
        ],
        "split": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "total": int(len(df)),
            "test_share": float(len(X_test) / len(df)),
        },
        "feature_importance": feature_importance,
        "fare_bands": _band_table(result, "fare_band", FARE_LABELS),
        "distance_bands": _band_table(result, "distance_band", DIST_LABELS),
        "residuals": {
            "mean": float(residuals.mean()),
            "median": float(residuals.median()),
            "p05": float(residuals.quantile(0.05)),
            "p10": float(residuals.quantile(0.10)),
            "p25": float(residuals.quantile(0.25)),
            "p75": float(residuals.quantile(0.75)),
            "p90": float(residuals.quantile(0.90)),
            "p95": float(residuals.quantile(0.95)),
            "within_2": float((residuals.abs() <= 2).mean()),
            "within_5": float((residuals.abs() <= 5).mean()),
            "within_10": float((residuals.abs() <= 10).mean()),
        },
        "model_card": {
            "algorithm": "LightGBM Regressor",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42,
            "target": "total_amount",
            "features": FEATURES,
            "feature_count": len(FEATURES),
        },
    }

    # A portable fallback for macOS machines where libomp is not installed.
    try:
        sample_idx = np.random.default_rng(7).choice(
            len(X_train_t), size=min(500_000, len(X_train_t)), replace=False
        )
        sample_idx = np.sort(sample_idx)
        X_sample = X_train_t.iloc[sample_idx]
        y_sample = y_train.iloc[sample_idx]
        fallback = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42,
            early_stopping=False,
        )
        fallback.fit(X_sample, y_sample)
        fallback_pred = fallback.predict(X_test_t)
        joblib.dump(fallback, MODELS / "taxi_fallback.joblib")
        payload["fallback"] = {
            "algorithm": "HistGradientBoostingRegressor",
            "mae": float(mean_absolute_error(y_test, fallback_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, fallback_pred))),
            "r2": float(r2_score(y_test, fallback_pred)),
            "training_rows": int(len(sample_idx)),
        }
        print("Fallback model trained.")
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"Fallback model not trained: {exc}")

    (DATA / "model_metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def build_fare_assets(df: pd.DataFrame) -> None:
    """Fare-shape aggregates from the model-eligible set."""
    work = df.copy()
    work["fare_per_mile"] = np.where(
        work["trip_distance"] > 0,
        work["fare_amount"] / work["trip_distance"],
        np.nan,
    )
    work["tip_rate"] = np.where(
        work["total_amount"] > 0, work["tip_amount"] / work["total_amount"], 0.0
    )

    work["distance_band"] = pd.cut(
        work["trip_distance"], bins=DIST_BINS, labels=DIST_LABELS, right=False
    )
    distance = (
        work.groupby("distance_band", observed=True)
        .agg(
            trips=("total_amount", "size"),
            avg_total=("total_amount", "mean"),
            median_total=("total_amount", "median"),
            avg_fare=("fare_amount", "mean"),
            avg_tip=("tip_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
            median_fare_per_mile=("fare_per_mile", "median"),
            avg_fare_per_mile=("fare_per_mile", "mean"),
        )
        .reindex(DIST_LABELS)
        .reset_index()
        .rename(columns={"distance_band": "band"})
    )
    distance["share"] = distance["trips"] / distance["trips"].sum()
    distance.to_csv(DATA / "distance_bands.csv", index=False)

    fare_hourly = (
        work.groupby("pickup_hour")
        .agg(
            trips=("total_amount", "size"),
            avg_total=("total_amount", "mean"),
            median_total=("total_amount", "median"),
            avg_fare=("fare_amount", "mean"),
            median_fare=("fare_amount", "median"),
            avg_fare_per_mile=("fare_per_mile", "mean"),
            median_fare_per_mile=("fare_per_mile", "median"),
            avg_tip=("tip_amount", "mean"),
            tip_rate=("tip_rate", "mean"),
        )
        .reindex(range(24))
        .reset_index()
    )
    fare_hourly["hour_label"] = fare_hourly["pickup_hour"].map(hour_label)
    fare_hourly.to_csv(DATA / "fare_hourly.csv", index=False)

    fare_dow = (
        work.groupby("pickup_day_of_week")
        .agg(
            trips=("total_amount", "size"),
            avg_total=("total_amount", "mean"),
            median_total=("total_amount", "median"),
            avg_fare=("fare_amount", "mean"),
            median_fare=("fare_amount", "median"),
            avg_fare_per_mile=("fare_per_mile", "mean"),
            median_fare_per_mile=("fare_per_mile", "median"),
            avg_tip=("tip_amount", "mean"),
            tip_rate=("tip_rate", "mean"),
        )
        .reindex(DAY_ORDER)
        .reset_index()
    )
    fare_dow.to_csv(DATA / "fare_dow.csv", index=False)


def build_model_sample(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(350_000, len(df)), random_state=42).copy()
    keep = [
        "tpep_pickup_datetime",
        "pickup_hour",
        "pickup_day",
        "pickup_day_of_week",
        "pickup_day_type",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "tip_amount",
        "tolls_amount",
        "extra",
        "mta_tax",
        "improvement_surcharge",
        "congestion_surcharge",
        "Airport_fee",
        "cbd_congestion_fee",
        "passenger_count",
        "payment_type",
        "RatecodeID",
        "VendorID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "pickup_year",
        "pickup_month",
        "duration_min",
        "fare_per_mile",
    ]
    keep = [c for c in keep if c in sample.columns]
    sample = sample[keep].rename(columns={"tpep_pickup_datetime": "pickup_datetime"})
    sample.to_parquet(DATA / "taxi_sample.parquet", index=False)


# ------------------------------------------------------------------- driver ---
def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    trips_path = RAW / "yellow_tripdata_2025-01.parquet"
    zone_path = RAW / "taxi_zone_lookup.csv"
    download(TRIPS_URL, trips_path)
    download(ZONE_URL, zone_path)

    # The lookup is a static dashboard asset; keep a copy in data/.
    zone_lookup = pd.read_csv(zone_path, encoding="utf-8-sig")
    zone_lookup.to_csv(DATA / "taxi_zone_lookup.csv", index=False)

    raw = pd.read_parquet(trips_path)
    print(f"Raw rows: {len(raw):,}")
    january_rows = int(
        (
            (raw["tpep_pickup_datetime"] >= "2025-01-01")
            & (raw["tpep_pickup_datetime"] < "2025-02-01")
        ).sum()
    )
    eda = clean_eda(raw)
    print(f"January rows: {january_rows:,}")
    print(f"EDA rows: {len(eda):,}")
    summary = build_eda_assets(eda)
    summary["january_trips"] = january_rows
    summary["excluded_trips"] = january_rows - len(eda)

    ml = model_eligible(eda)
    summary["model_eligible_trips"] = int(len(ml))
    summary["model_excluded_trips"] = int(len(eda) - len(ml))
    (DATA / "summary_eda.json").write_text(json.dumps(summary, indent=2))
    build_fare_assets(ml)
    build_model_sample(ml)
    model_payload = build_model_assets(ml)
    if model_payload:
        summary["model"] = model_payload["metrics"]
        summary["model_test_rows"] = model_payload["split"]["test"]
        (DATA / "summary.json").write_text(json.dumps(summary, indent=2))
    else:
        (DATA / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"EDA trips       : {len(eda):,}")
    print(f"Model-eligible  : {len(ml):,}")
    print(f"Revenue         : ${summary['revenue']:,.0f}")
    print(f"Peak hour       : {summary['peak_hour']:02d}:00 · {summary['peak_hour_trips']:,} trips")
    print(f"Busiest day     : Jan {summary['peak_day']} · {summary['peak_day_trips']:,} trips")
    print(f"Assets written  : {DATA.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
