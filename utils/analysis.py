"""Shared calculations and formatting for the taxi dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

RATE_LABELS = {
    1.0: "Standard meter",
    2.0: "JFK flat fare",
    3.0: "Newark",
    4.0: "Nassau / Westchester",
    5.0: "Negotiated",
    6.0: "Group ride",
    99.0: "Unknown",
}

PAYMENT_LABELS = {
    0: "Unknown",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Voided",
    6: "Unknown",
}

FARE_BINS = [0, 15, 25, 40, 70, 1_000]
FARE_LABELS = ["<$15", "$15–25", "$25–40", "$40–70", "$70+"]
DIST_BINS = [0, 1, 2, 3, 5, 10, 20, 101]
DIST_LABELS = ["<1 mi", "1–2 mi", "2–3 mi", "3–5 mi", "5–10 mi", "10–20 mi", "20+ mi"]


# ------------------------------------------------------------------ format ---
def num(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


def money(value: float, *, digits: int = 0, compact: bool = False) -> str:
    if compact:
        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:,.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:,.1f}M"
        if abs(value) >= 1_000:
            return f"${value / 1_000:,.0f}k"
    return f"${value:,.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def miles(value: float) -> str:
    return f"{value:,.1f} mi"


def hour_label(hour: int) -> str:
    if hour == 0:
        return "12 AM"
    if hour < 12:
        return f"{hour} AM"
    if hour == 12:
        return "12 PM"
    return f"{hour - 12} PM"


def hour_short(hour: int) -> str:
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


def rate_label(value: float) -> str:
    return RATE_LABELS.get(float(value), "Unknown")


def payment_label(value: int) -> str:
    return PAYMENT_LABELS.get(int(value), "Unknown")


def trip_tier(distance: float) -> str:
    if distance < 2:
        return "Short hop"
    if distance <= 10:
        return "City transit"
    return "Long haul / airport"


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+,.{digits}f}"


# --------------------------------------------------------------- summaries ---
def top_zones(
    zones: pd.DataFrame,
    *,
    kind: str = "pickup",
    metric: str = "trips",
    n: int = 10,
    min_trips: int = 50,
) -> pd.DataFrame:
    """Return a ranked zone table with a display label."""
    prefix = "pickup" if kind == "pickup" else "dropoff"
    trips_col = f"{prefix}_trips"
    metric_cols = {
        "trips": trips_col,
        "revenue": f"{prefix}_revenue",
        "avg_total": f"{prefix}_avg_total",
        "avg_distance": f"{prefix}_avg_distance",
    }
    value_col = metric_cols[metric]
    table = zones[zones[trips_col] >= min_trips].copy()
    table = table.sort_values(value_col, ascending=False).head(n).reset_index(drop=True)
    table["label"] = table["Zone"].str.replace(r"\s*/\s*", " / ", regex=True)
    table["short_label"] = table["label"].apply(
        lambda s: s if len(s) <= 31 else s[:29].rstrip() + "…"
    )
    table["share"] = table[trips_col] / max(zones[trips_col].sum(), 1)
    return table


def zone_metric(zones: pd.DataFrame, metric: str) -> tuple[pd.Series, str, str]:
    """Return map values, label and hover formatting for a zone metric."""
    if metric == "revenue":
        values = zones["pickup_revenue"]
        return values, "Pickup revenue", "$"
    if metric == "avg_total":
        values = zones["pickup_avg_total"]
        return values, "Average total", "$"
    return zones["pickup_trips"], "Pickup trips", ""


def airport_rows(ratecode: pd.DataFrame) -> pd.DataFrame:
    keep = [1.0, 2.0, 3.0, 4.0, 5.0]
    table = ratecode[ratecode["RatecodeID"].isin(keep)].copy()
    order = {value: i for i, value in enumerate(keep)}
    table["order"] = table["RatecodeID"].map(order)
    return table.sort_values("order").drop(columns="order").reset_index(drop=True)


def residual_range(metrics: dict, predicted: float) -> tuple[float, float]:
    """Pick the residual interval for the fare band closest to a prediction."""
    bands = metrics.get("fare_bands", [])
    if not bands:
        p10, p90 = metrics["residuals"]["p10"], metrics["residuals"]["p90"]
        return max(0.0, predicted + p10), predicted + p90
    labels = {"<$15": 0, "$15–25": 1, "$25–40": 2, "$40–70": 3, "$70+": 4}
    if predicted < 15:
        chosen = bands[0]
    elif predicted < 25:
        chosen = bands[1]
    elif predicted < 40:
        chosen = bands[2]
    elif predicted < 70:
        chosen = bands[3]
    else:
        chosen = bands[4]
    return max(0.0, predicted + chosen["p10"]), predicted + chosen["p90"]
