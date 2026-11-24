"""Insights — a single briefing on demand, fares, the model and the estimate."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import analysis as an  # noqa: E402
from utils import data_loader, theme  # noqa: E402

summary = data_loader.load_summary()
metrics = data_loader.load_model_metrics()
ratecode = data_loader.load_table("ratecode.csv")
distance_bands = data_loader.load_table("distance_bands.csv")

m = metrics["metrics"]
r = metrics["residuals"]
importance = {row["feature"]: row for row in metrics["feature_importance"]}
fare_bands = metrics["fare_bands"]
zone_share = importance["Pickup zone"]["share"] + importance["Drop-off zone"]["share"]
jfk = ratecode.loc[ratecode["rate_label"] == "JFK flat fare"].iloc[0]
newark = ratecode.loc[ratecode["rate_label"] == "Newark"].iloc[0]
short_hop = distance_bands.iloc[0]
long_haul = distance_bands.iloc[-1]

theme.header(
    "The month, <em>in brief</em>.",
    "The patterns worth carrying into a dispatch shift, a fare conversation or a "
    "model review. Every signal comes from the same January 2025 trip set; none of "
    "it is a forecast.",
    pills=[
        f"<b>{summary['total_trips'] / 1e6:.2f}M</b> valid trips",
        f"<b>{summary['peak_hour_trips']:,}</b> trips at 6 PM",
        f"<b>{an.money(summary['avg_total'], digits=2)}</b> average total",
        f"<b>{m['r2']:.3f}</b> holdout R²",
    ],
    icon="insights",
)

theme.section(
    "When the city moves",
    "Demand is a clock. The morning builds, the evening peaks, and the weekend "
    "keeps a later schedule.",
    kicker="DEMAND",
)
theme.briefing(
    [
        {
            "tag": "THE EVENING PEAK",
            "stat": f"{summary['peak_hour_trips']:,}",
            "title": "trips in the 6 PM hour",
            "copy": "The evening peak also earns "
            f"{an.money(summary['peak_hour_revenue'], compact=True)} in meter revenue "
            "in that hour alone. It is a narrow window, not a plateau.",
            "accent": theme.YELLOW,
        },
        {
            "tag": "TWO CLOCKS",
            "stat": f"{summary['weekday_share']:.0%}",
            "title": "of trips run Monday–Friday",
            "copy": "Weekend demand stays high after midnight while weekday demand "
            "falls away. The same fleet works a different shift on Saturday and Sunday.",
            "accent": theme.TEAL,
        },
        {
            "tag": "THE OUTLIER DAY",
            "stat": f"Jan {summary['peak_day']}",
            "title": "the busiest and richest day",
            "copy": f"{summary['peak_day_trips']:,} trips and "
            f"{an.money(summary['peak_day_revenue'], compact=True)} in meter revenue. "
            "Mid-month and midweek, it is the clearest single-day spike in January.",
            "accent": theme.ORANGE,
        },
        {
            "tag": "THE 5 AM PREMIUM",
            "stat": an.money(summary["highest_avg_fare"], digits=2),
            "title": "the highest average fare of the day",
            "copy": "Fewer trips, longer airport and outer-borough runs. The early "
            "hour pays for the empty streets.",
            "accent": theme.YELLOW,
        },
    ]
)

theme.section(
    "What the meter charges",
    "Distance sets the scale. The premium comes from airports, short hops and the "
    "places where supply meets demand.",
    kicker="FARES",
)
theme.briefing(
    [
        {
            "tag": "THE AIRPORT PREMIUM",
            "stat": f"{summary['airport_share']:.2%}",
            "title": "of trips are JFK or Newark runs",
            "copy": f"Those trips produce {summary['airport_revenue_share']:.1%} of all "
            "meter revenue. Average total: "
            f"{an.money(jfk['avg_total'], digits=2)} at JFK and "
            f"{an.money(newark['avg_total'], digits=2)} at Newark.",
            "accent": theme.ORANGE,
        },
        {
            "tag": "THE DISTANCE LADDER",
            "stat": f"{an.money(short_hop['avg_total'], digits=2)} → "
            f"{an.money(long_haul['avg_total'], digits=2)}",
            "title": "average total, under 1 mile to 20+ miles",
            "copy": "The fare is not linear. Base charges and fees travel with every "
            "trip, so the shortest rides carry the highest fixed-cost share.",
            "accent": theme.YELLOW,
        },
        {
            "tag": "THE SHORT-HOP PREMIUM",
            "stat": f"{an.money(short_hop['median_fare_per_mile'], digits=2)} / mi",
            "title": "median fare per mile under one mile",
            "copy": f"At 20+ miles the median falls to "
            f"{an.money(long_haul['median_fare_per_mile'], digits=2)} per mile. Short "
            "trips pay for the meter's time.",
            "accent": theme.TEAL,
        },
    ]
)

theme.section(
    "How the fare engine scores",
    "Strong overall fit, with the largest misses on expensive airport and long-haul "
    "trips. That is the right place to be cautious.",
    kicker="MODEL",
)
theme.briefing(
    [
        {
            "tag": "DISTANCE LEADS",
            "stat": f"{importance['Trip distance']['share']:.0%}",
            "title": "of tree splits use trip distance",
            "copy": "The model learns the meter's first principle before it learns "
            "the city. Distance is the strongest single driver by a clear margin.",
            "accent": theme.YELLOW,
        },
        {
            "tag": "ZONES CARRY CONTEXT",
            "stat": f"{zone_share:.0%}",
            "title": "of splits use the pickup or drop-off zone",
            "copy": "Location is how the model sees airports, Manhattan short hops "
            "and outer-borough runs.",
            "accent": theme.ORANGE,
        },
        {
            "tag": "ERROR GROWS WITH FARE",
            "stat": f"{an.money(fare_bands[0]['mae'], digits=2)} → "
            f"{an.money(fare_bands[-1]['mae'], digits=2)}",
            "title": "MAE from the under-$15 band to the $70+ band",
            "copy": f"{r['within_5']:.1%} of holdout errors are within $5, and "
            f"{r['within_2']:.1%} are within $2. The model is sharpest where most "
            "trips live.",
            "accent": theme.TEAL,
        },
    ]
)

theme.section(
    "How to use the estimate",
    "The model is most useful as a quick prior. The comparison group tells you "
    "whether the estimate is plausible for this kind of trip.",
    kicker="ESTIMATOR",
)
theme.briefing(
    [
        {
            "tag": "WHAT MOVES THE NUMBER",
            "stat": "Distance",
            "title": "the first dial",
            "copy": "Distance is the strongest driver, followed by drop-off zone, "
            "rate code, pickup hour and payment type. The model learns the shape of "
            "the meter.",
            "accent": theme.YELLOW,
        },
        {
            "tag": "WHAT IT CANNOT SEE",
            "stat": "Traffic",
            "title": "the variables the model cannot see",
            "copy": "Wait time, route choice, weather and driver behaviour are not in "
            "the feature set. Treat the range as a useful prior, not a fare guarantee.",
            "accent": theme.ORANGE,
        },
        {
            "tag": "WHY THE RANGE MOVES",
            "stat": f"±{an.money(m['mae'], digits=2)}",
            "title": "typical error",
            "copy": "The range widens for expensive trips because the model's misses "
            "widen there too. The result card uses the error band that matches the "
            "predicted fare.",
            "accent": theme.TEAL,
        },
        {
            "tag": "THE REALITY CHECK",
            "stat": "350K",
            "title": "similar trips in the comparison sample",
            "copy": "The estimator compares each result with trips that share a day "
            "type, rate class, nearby hour and distance. The observed median keeps "
            "the model honest.",
            "accent": theme.YELLOW,
        },
    ]
)

theme.footer()
