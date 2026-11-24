"""Feature construction and fare estimation for the saved taxi model."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_ORDER = [
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

PAYMENT_CODES = {
    "Credit card": 1,
    "Cash": 2,
    "No charge": 3,
    "Dispute": 4,
}

RATE_OPTIONS = {
    "Standard meter": 1.0,
    "JFK flat fare": 2.0,
    "Newark": 3.0,
    "Nassau / Westchester": 4.0,
    "Negotiated": 5.0,
}

VENDOR_ID = 1
STORE_AND_FWD_FLAG = 0
PICKUP_YEAR = 2025
PICKUP_MONTH = 1
PICKUP_DAY = 15


def build_input_df(
    *,
    trip_distance: float,
    passenger_count: int,
    pickup_hour: int,
    day_of_week: str,
    pu_id: int,
    do_id: int,
    payment_label: str,
    rate_label: str,
) -> pd.DataFrame:
    """Build one raw row in the exact training schema."""
    day_type = "Weekend" if day_of_week in ("Saturday", "Sunday") else "Weekday"
    row = {
        "VendorID": VENDOR_ID,
        "passenger_count": int(passenger_count),
        "trip_distance": float(trip_distance),
        "RatecodeID": RATE_OPTIONS.get(rate_label, 1.0),
        "store_and_fwd_flag": STORE_AND_FWD_FLAG,
        "PULocationID": int(pu_id),
        "DOLocationID": int(do_id),
        "payment_type": PAYMENT_CODES.get(payment_label, 1),
        "pickup_year": PICKUP_YEAR,
        "pickup_month": PICKUP_MONTH,
        "pickup_day": PICKUP_DAY,
        "pickup_hour": int(pickup_hour),
        "pickup_day_of_week": day_of_week,
        "pickup_day_type": day_type,
    }
    return pd.DataFrame([{key: row[key] for key in FEATURE_ORDER}])


def predict_fare(model, preprocessor, input_df: pd.DataFrame) -> float:
    encoded = preprocessor.transform(input_df)
    pred = float(model.predict(encoded)[0])
    return max(0.0, pred)


def _hour_gap(a: pd.Series, hour: int) -> pd.Series:
    diff = (a - hour).abs()
    return np.minimum(diff, 24 - diff)


def similar_trips(
    sample: pd.DataFrame,
    *,
    trip_distance: float,
    pickup_hour: int,
    day_of_week: str,
    rate_label: str,
    max_rows: int = 1_200,
) -> tuple[pd.DataFrame, dict]:
    """Find observed trips that look like the requested ride.

    The comparison group is intentionally simple: same day type, same rate
    class, nearby hour and nearby distance. It is a sanity check, not a model
    feature.
    """
    day_type = "Weekend" if day_of_week in ("Saturday", "Sunday") else "Weekday"
    rate_code = RATE_OPTIONS.get(rate_label, 1.0)
    if rate_code in (2.0, 3.0, 4.0, 5.0):
        rate_mask = sample["RatecodeID"] == rate_code
    else:
        rate_mask = sample["RatecodeID"] == 1.0

    base = sample[
        rate_mask
        & (sample["pickup_day_type"] == day_type)
        & (sample["trip_distance"] > 0)
    ]
    if base.empty:
        base = sample[sample["trip_distance"] > 0]

    distance_tol = max(1.0, trip_distance * 0.25)
    hour_tol = 2
    for dist_tol, h_tol in (
        (distance_tol, hour_tol),
        (max(2.0, distance_tol * 1.6), 3),
        (max(3.0, distance_tol * 2.2), 5),
    ):
        subset = base[
            (base["trip_distance"].between(trip_distance - dist_tol, trip_distance + dist_tol))
            & (_hour_gap(base["pickup_hour"], pickup_hour) <= h_tol)
        ]
        if len(subset) >= 40:
            break

    if subset.empty:
        subset = base[
            base["trip_distance"].between(
                max(0.1, trip_distance - max(2.0, distance_tol * 2)),
                trip_distance + max(2.0, distance_tol * 2),
            )
        ]

    if subset.empty:
        summary = {
            "count": 0,
            "median": 0.0,
            "mean": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "avg_distance": 0.0,
        }
        return subset.head(0), summary

    if len(subset) > max_rows:
        subset = subset.sample(max_rows, random_state=42)

    values = subset["total_amount"]
    summary = {
        "count": int(len(subset)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "avg_distance": float(subset["trip_distance"].mean()),
    }
    return subset, summary
