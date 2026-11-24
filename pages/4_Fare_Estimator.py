"""Fare Estimator — a live meter for a hypothetical January taxi trip."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import analysis as an  # noqa: E402
from utils import data_loader, modeling, theme  # noqa: E402

DAY_OPTIONS = an.DAY_ORDER
PAYMENT_OPTIONS = list(modeling.PAYMENT_CODES)
RATE_OPTIONS = list(modeling.RATE_OPTIONS)

DEFAULTS = {
    "preset": "Custom",
    "pickup_id": 161,
    "dropoff_id": 237,
    "distance": 1.4,
    "passengers": 2,
    "hour": 18,
    "day": "Friday",
    "payment": "Credit card",
    "rate": "Standard meter",
}

PRESETS = {
    "Midtown hop": {
        "pickup_id": 161,
        "dropoff_id": 237,
        "distance": 1.4,
        "passengers": 2,
        "hour": 13,
        "day": "Wednesday",
        "payment": "Credit card",
        "rate": "Standard meter",
    },
    "JFK arrival": {
        "pickup_id": 132,
        "dropoff_id": 161,
        "distance": 16.0,
        "passengers": 2,
        "hour": 8,
        "day": "Friday",
        "payment": "Credit card",
        "rate": "JFK flat fare",
    },
    "Late-night loop": {
        "pickup_id": 230,
        "dropoff_id": 79,
        "distance": 2.8,
        "passengers": 3,
        "hour": 1,
        "day": "Saturday",
        "payment": "Cash",
        "rate": "Standard meter",
    },
    "LaGuardia business": {
        "pickup_id": 138,
        "dropoff_id": 161,
        "distance": 8.5,
        "passengers": 1,
        "hour": 7,
        "day": "Monday",
        "payment": "Credit card",
        "rate": "Standard meter",
    },
}

zones = data_loader.load_zones()
summary = data_loader.load_summary()
metrics = data_loader.load_model_metrics()
sample = data_loader.load_sample()

zone_labels = {
    int(row["LocationID"]): f"{int(row['LocationID'])} · {row['Zone']}, {row['Borough']}"
    for _, row in zones.sort_values("LocationID").iterrows()
}
label_to_id = {label: zone_id for zone_id, label in zone_labels.items()}


def label_for(zone_id: int) -> str:
    return zone_labels.get(int(zone_id), next(iter(zone_labels.values())))


def apply_preset() -> None:
    preset = st.session_state.get("preset", "Custom")
    if preset == "Custom":
        return
    values = PRESETS.get(preset, {})
    for key, value in values.items():
        if key in ("pickup_id", "dropoff_id"):
            st.session_state[f"{key}_label"] = label_for(value)
        else:
            st.session_state[key] = value
    st.session_state.pop("estimate_result", None)


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
for zone_id in (st.session_state.pickup_id, st.session_state.dropoff_id):
    label_key = "pickup_id_label" if zone_id == st.session_state.pickup_id else "dropoff_id_label"
    if label_key not in st.session_state:
        st.session_state[label_key] = label_for(zone_id)

try:
    model, preprocessor, model_label, fallback = data_loader.load_artifacts()
    artifact_error = None
except Exception as exc:  # pragma: no cover - deployment should have the artifact
    model = preprocessor = None
    model_label, fallback = "Unavailable", False
    artifact_error = str(exc)

theme.header(
    "Price the <em>ride</em>.",
    "Set a pickup, a drop-off and a time. The saved fare model prices the total, "
    "then the dashboard shows what similar trips actually paid.",
    pills=[
        f"<b>{an.money(metrics['metrics']['mae'], digits=2)}</b> typical error",
        f"<b>{metrics['metrics']['r2']:.3f}</b> holdout R²",
        f"<b>{metrics['residuals']['within_5']:.1%}</b> within $5",
        "Total amount · tips, tolls and fees included",
    ],
    icon="estimator",
)

if artifact_error:
    st.error(
        "The fare model could not be loaded in this environment. "
        "Run the dashboard on a machine with LightGBM and `libomp`, or keep the "
        "portable fallback model in `models/taxi_fallback.joblib`."
    )
    st.caption(artifact_error)
    theme.footer()
    st.stop()

st.html(
    theme.chips(
        [
            ("14 FEATURES", "dark"),
            ("NO FARE LEAKAGE", "yellow"),
            ("JAN 2025 SCHEMA", "teal"),
        ]
    )
)

st.selectbox(
    "Start with a preset",
    ["Custom", "Midtown hop", "JFK arrival", "Late-night loop", "LaGuardia business"],
    key="preset",
    on_change=apply_preset,
    help="Presets fill the form with a familiar trip. Change anything afterwards.",
)

with st.form("fare_form", border=True):
    st.html(
        "<div style='font-family:var(--display);font-size:26px;color:var(--asphalt);"
        "letter-spacing:.03em;text-transform:uppercase;line-height:1;'>Trip details</div>"
        "<div style='font-size:11.5px;color:var(--muted);margin:2px 0 12px;'>"
        "The model was trained on January 2025 trips; the date is held at a "
        "mid-month Wednesday for consistency.</div>"
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        pickup_label = st.selectbox(
            "Pickup zone",
            options=list(label_to_id),
            key="pickup_id_label",
        )
    with c2:
        dropoff_label = st.selectbox(
            "Drop-off zone",
            options=list(label_to_id),
            key="dropoff_id_label",
        )
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        distance = st.slider(
            "Trip distance (miles)",
            min_value=0.2,
            max_value=40.0,
            step=0.1,
            key="distance",
        )
    with c2:
        passengers = st.selectbox(
            "Passengers", [1, 2, 3, 4, 5, 6], key="passengers"
        )
    with c3:
        hour = st.selectbox(
            "Pickup hour",
            list(range(24)),
            format_func=an.hour_label,
            key="hour",
        )
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        day = st.selectbox("Day", DAY_OPTIONS, key="day")
    with c2:
        payment = st.selectbox("Payment", PAYMENT_OPTIONS, key="payment")
    with c3:
        rate = st.selectbox(
            "Rate code",
            RATE_OPTIONS,
            key="rate",
            help="JFK, Newark and Nassau/Westchester use special rate codes.",
        )
    submitted = st.form_submit_button(
        "Run the meter", type="primary", use_container_width=True
    )

if submitted:
    pu_id = label_to_id[pickup_label]
    do_id = label_to_id[dropoff_label]
    payload = modeling.build_input_df(
        trip_distance=distance,
        passenger_count=passengers,
        pickup_hour=hour,
        day_of_week=day,
        pu_id=pu_id,
        do_id=do_id,
        payment_label=payment,
        rate_label=rate,
    )
    fare = modeling.predict_fare(model, preprocessor, payload)
    low, high = an.residual_range(metrics, fare)
    _, comps_summary = modeling.similar_trips(
        sample,
        trip_distance=distance,
        pickup_hour=hour,
        day_of_week=day,
        rate_label=rate,
    )
    st.session_state["estimate_result"] = {
        "fare": fare,
        "low": low,
        "high": high,
        "per_mile": fare / distance if distance else 0.0,
        "tier": an.trip_tier(distance),
        "rate": rate,
        "distance": distance,
        "hour": hour,
        "day": day,
        "comps_summary": comps_summary,
        "model_label": model_label,
        "fallback": fallback,
        "pickup": pickup_label,
        "dropoff": dropoff_label,
        "payment": payment,
    }

result = st.session_state.get("estimate_result")

if not result:
    theme.empty_state()
    st.html(
        "<div style='margin-top:12px;font-size:11.5px;color:var(--muted);"
        "line-height:1.65;'>"
        "<b style='color:var(--asphalt);'>How it works.</b> The model uses trip "
        "distance, pickup and drop-off zones, time, rate code, payment type and "
        "passenger count. It predicts the total amount, not the base fare. "
        "The comparison group is built from the 350,000-trip sample."
        "</div>"
    )
else:
    comps_summary = result["comps_summary"]
    theme.meter_card(
        result["fare"],
        result["low"],
        result["high"],
        result["per_mile"],
        result["tier"],
        result["rate"],
        result["distance"],
        an.hour_label(result["hour"]),
        result["day"],
        comps_summary,
        model_label=result["model_label"],
        fallback=result["fallback"],
    )
    st.html(
        "<div style='font-size:10.5px;color:var(--muted);line-height:1.6;'>"
        f"<b style='color:var(--asphalt);'>Trip:</b> {result['pickup']} → "
        f"{result['dropoff']} · {an.money(result['per_mile'], digits=2)} per mile · "
        f"{result['payment']}. The estimate is a model output, not a quote."
        "</div>"
    )

theme.footer()
