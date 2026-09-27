"""Taxi Fare & Trip Price Estimator — modular Streamlit frontend.

Loads a pre-trained regressor (`taxi_model.joblib`) and its fitted
ColumnTransformer (`taxi_preprocessor.joblib`) from the current directory
and estimates the trip fare in USD from raw trip attributes.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = "taxi_model.joblib"
PREPROCESSOR_PATH = "taxi_preprocessor.joblib"

# Measured on a reference trip: the model outputs dollars directly
# (3.5 mi -> ~$32.7), NOT log-dollars. Keep False unless retrained
# on a log-transformed target (e.g. log_fare).
LOG_TRANSFORMED_TARGET = False

# Exact training column order: numerical_features + categorical_features
# (see taxi.ipynb). Order is cosmetic here (the transformer resolves by
# name), but matching it keeps the pipeline explicit.
FEATURE_ORDER = [
    "VendorID", "passenger_count", "trip_distance", "RatecodeID",
    "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type",
    "pickup_year", "pickup_month", "pickup_day", "pickup_hour",
    "pickup_day_of_week", "pickup_day_type",
]

# TLC payment_type codes.
PAYMENT_MAP = {"Credit Card": 1, "Cash": 2, "No Charge": 3, "Dispute": 4}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday"]

# Fixed trip-context defaults (not exposed as inputs).
VENDOR_ID = 1
STORE_AND_FWD_FLAG = 0
PICKUP_DAY, PICKUP_MONTH, PICKUP_YEAR = 15, 6, 2026
RATECODE_ID = 1

PRESETS = {
    "Short City Hopper": {
        "trip_distance": 1.2, "passenger_count": 2, "pickup_hour": 13,
        "day_of_week": "Wednesday", "pu_id": 161, "do_id": 237,
        "payment_type": "Credit Card",
    },
    "Airport Trip": {
        "trip_distance": 18.0, "passenger_count": 2, "pickup_hour": 7,
        "day_of_week": "Friday", "pu_id": 100, "do_id": 132,
        "payment_type": "Credit Card",
    },
    "Late Night Weekend": {
        "trip_distance": 4.5, "passenger_count": 3, "pickup_hour": 1,
        "day_of_week": "Saturday", "pu_id": 90, "do_id": 100,
        "payment_type": "Cash",
    },
}


# ------------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Load regressor + preprocessor once; raises FileNotFoundError."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


# ------------------------------------------------------------------
# Preset handling
# ------------------------------------------------------------------
def apply_preset():
    """Sidebar callback: copy the chosen preset into widget state."""
    name = st.session_state.get("preset_name", "Custom")
    if name in PRESETS:
        for key, value in PRESETS[name].items():
            st.session_state[key] = value


# ------------------------------------------------------------------
# Prediction pipeline
# ------------------------------------------------------------------
def build_input_df(trip_distance, passenger_count, pickup_hour, day_of_week,
                   pu_id, do_id, payment_label):
    """Single-row raw DataFrame in exact training schema + memory-safe dtypes.

    int8 covers all small ints; pickup_year (2026) needs int16.
    Categorical pair is cast to `category` for the OneHotEncoder.
    """
    day_type = "Weekend" if day_of_week in ("Saturday", "Sunday") else "Weekday"
    row = {
        "VendorID": VENDOR_ID,
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
        "RatecodeID": RATECODE_ID,
        "store_and_fwd_flag": STORE_AND_FWD_FLAG,
        "PULocationID": pu_id,
        "DOLocationID": do_id,
        "payment_type": PAYMENT_MAP[payment_label],
        "pickup_year": PICKUP_YEAR,
        "pickup_month": PICKUP_MONTH,
        "pickup_day": PICKUP_DAY,
        "pickup_hour": pickup_hour,
        "pickup_day_of_week": day_of_week,
        "pickup_day_type": day_type,
    }
    df = pd.DataFrame([{k: row[k] for k in FEATURE_ORDER}])
    return df.astype({
        "VendorID": "int8", "passenger_count": "int8",
        "trip_distance": "float32", "RatecodeID": "int8",
        "store_and_fwd_flag": "int8", "PULocationID": "int16",
        "DOLocationID": "int16", "payment_type": "int8",
        "pickup_year": "int16", "pickup_month": "int8",
        "pickup_day": "int8", "pickup_hour": "int8",
        "pickup_day_of_week": "category", "pickup_day_type": "category",
    })


def estimate_fare(input_df, model, preprocessor):
    """Transform raw row, predict, and invert log-target if configured."""
    pred = float(model.predict(preprocessor.transform(input_df))[0])
    if LOG_TRANSFORMED_TARGET:
        pred = float(np.expm1(pred))
    return max(0.0, pred)


def trip_tier(trip_distance):
    if trip_distance < 2.0:
        return "Short Trip"
    if trip_distance <= 10.0:
        return "Standard Transit"
    return "Long Distance / Airport"


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Taxi Fare & Trip Price Estimator",
        page_icon="🚖",
        layout="wide",
    )

    try:
        model, preprocessor = load_assets()
    except FileNotFoundError as err:
        st.error(
            "Model files not found. Make sure "
            f"'{MODEL_PATH}' and '{PREPROCESSOR_PATH}' are in the same "
            f"folder as app.py. (Details: {err})"
        )
        st.stop()
    except Exception as err:
        st.error(f"Could not load model assets: {err}")
        st.stop()

    st.title("🚖 Taxi Fare & Trip Price Estimator")
    st.write(
        "Describe your trip below (or pick a preset) and get an instant "
        "LightGBM fare estimate in USD."
    )

    with st.sidebar:
        st.header("Trip Presets")
        st.selectbox(
            "Pre-fill Preset Trips",
            ["Custom"] + list(PRESETS),
            key="preset_name",
            on_change=apply_preset,
        )
        st.caption("Choosing a preset fills the trip details automatically.")

    with st.container(border=True):
        st.subheader("Trip Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            trip_distance = st.slider(
                "Trip Distance (miles)", 0.1, 50.0, 2.5, 0.1,
                key="trip_distance",
            )
            passenger_count = st.selectbox(
                "Passengers", [1, 2, 3, 4, 5, 6], key="passenger_count",
            )
        with c2:
            pickup_hour = st.slider(
                "Pickup Hour (0-23)", 0, 23, 14, key="pickup_hour",
            )
            day_of_week = st.selectbox(
                "Day of Week", DAYS, index=DAYS.index("Friday"),
                key="day_of_week",
            )
        with c3:
            pu_id = st.number_input(
                "Pickup Zone ID", 1, 265, 100, key="pu_id",
            )
            do_id = st.number_input(
                "Dropoff Zone ID", 1, 265, 100, key="do_id",
            )
            payment_label = st.selectbox(
                "Payment Type", list(PAYMENT_MAP), key="payment_type",
            )

        estimate = st.button("Estimate Fare", type="primary")

    if estimate:
        try:
            input_df = build_input_df(
                float(trip_distance), int(passenger_count),
                int(pickup_hour), day_of_week,
                int(pu_id), int(do_id), payment_label,
            )
            fare = estimate_fare(input_df, model, preprocessor)
        except Exception as err:
            st.error(f"Error during prediction: {err}")
        else:
            st.divider()
            fare_col, break_col = st.columns([1.0, 1.2])
            with fare_col:
                st.metric("Estimated Fare", f"${fare:.2f}")
            with break_col:
                per_mile = fare / float(trip_distance) if trip_distance else 0.0
                st.markdown(
                    "**Fare breakdown**\n"
                    f"- Per-mile rate: **${per_mile:.2f}/mi**\n"
                    f"- Trip tier: **{trip_tier(float(trip_distance))}**\n"
                    f"- Passengers: **{passenger_count}** · "
                    f"Payment: **{payment_label}**"
                )


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------
# Setup / deployment
# ------------------------------------------------------------------
# pip install streamlit pandas numpy joblib scikit-learn lightgbm
# streamlit run app.py
