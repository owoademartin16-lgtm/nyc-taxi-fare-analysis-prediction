"""NYC Yellow Taxi fare intelligence — Streamlit dashboard.

Run with:

    python3 -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="NYC Taxi · Fare Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import analysis as an  # noqa: E402
from utils import charts, data_loader, theme  # noqa: E402

theme.inject()

try:
    st.logo(str(data_loader.ROOT / "assets" / "logo.svg"), size="large")
except Exception:
    pass


def overview() -> None:
    summary = data_loader.load_summary()
    hourly = data_loader.load_table("hourly.csv")
    daily = data_loader.load_table("daily.csv")
    daytype = data_loader.load_table("daytype.csv")
    zones = data_loader.load_zones()
    passengers = data_loader.load_table("passengers.csv")

    theme.header(
        "A month in the <em>back seat</em>.",
        "January 2025 through the windshield of New York's yellow taxis: where "
        "the city moved, when the meter ran, and what a ride actually costs.",
        pills=[
            f"<b>{summary['total_trips'] / 1e6:.2f}M</b> valid of "
            f"{summary['january_trips'] / 1e6:.2f}M January trips",
            f"<b>{an.money(summary['revenue'], compact=True)}</b> meter revenue",
            f"<b>{summary['peak_hour_trips']:,}</b> trips at 6 PM",
            f"<b>{summary['solo_share']:.0%}</b> solo riders",
        ],
        icon="dashboard",
    )

    theme.kpis(
        [
            {
                "label": "Valid trips",
                "value": f"{summary['total_trips'] / 1e6:.2f}M",
                "sub": f"of {summary['january_trips']:,} January trips",
                "accent": theme.YELLOW,
            },
            {
                "label": "Meter revenue",
                "value": an.money(summary["revenue"], compact=True),
                "sub": f"{an.money(summary['avg_total'], digits=2)} average total",
                "accent": theme.ORANGE,
            },
            {
                "label": "Peak hour",
                "value": an.hour_label(summary["peak_hour"]),
                "sub": f"{summary['peak_hour_trips']:,} trips · {an.money(summary['peak_hour_revenue'], compact=True)}",
                "accent": theme.YELLOW,
            },
            {
                "label": "Busiest day",
                "value": f"Jan {summary['peak_day']}",
                "sub": f"{summary['peak_day_trips']:,} trips · {an.money(summary['peak_day_revenue'], compact=True)}",
                "accent": theme.TEAL,
            },
            {
                "label": "Solo riders",
                "value": f"{summary['solo_share']:.0%}",
                "sub": f"1 passenger · {summary['median_distance']:.1f} mi median trip",
                "accent": theme.ASPHALT,
            },
            {
                "label": "Weekday share",
                "value": f"{summary['weekday_share']:.0%}",
                "sub": f"{summary['weekday_trips'] / 1e6:.2f}M vs {summary['weekend_trips'] / 1e6:.2f}M weekend",
                "accent": theme.ORANGE,
            },
        ],
        columns=6,
    )

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True, height=388):
            st.plotly_chart(
                charts.hourly_demand(hourly, height=340),
                config=theme.PLOTLY_CONFIG,
                use_container_width=True,
            )
    with right:
        with st.container(border=True, height=388):
            st.plotly_chart(
                charts.hourly_revenue(hourly, height=340),
                config=theme.PLOTLY_CONFIG,
                use_container_width=True,
            )

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True, height=388):
            st.plotly_chart(
                charts.daily_trips(daily, height=340),
                config=theme.PLOTLY_CONFIG,
                use_container_width=True,
            )
    with right:
        with st.container(border=True, height=388):
            st.plotly_chart(
                charts.daytype_donut(daytype, height=340),
                config=theme.PLOTLY_CONFIG,
                use_container_width=True,
            )

    left, right = st.columns([1.08, 0.92], gap="medium")
    with left:
        with st.container(border=True, height=436):
            st.plotly_chart(
                charts.top_zones_chart(
                    an.top_zones(zones, kind="pickup", metric="trips", n=10),
                    metric="trips",
                    kind="pickup",
                    height=378,
                ),
                config=theme.PLOTLY_CONFIG,
                use_container_width=True,
            )
    with right:
        with st.container(border=True, height=436):
            st.plotly_chart(
                charts.passenger_mix(passengers, height=378),
                config=theme.PLOTLY_CONFIG,
                use_container_width=True,
            )

    theme.footer()


pages = [
    st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/1_Demand_and_Time.py", title="Demand & Time", icon=":material/schedule:"),
    st.Page("pages/2_Fares_and_Zones.py", title="Fares & Zones", icon=":material/payments:"),
    st.Page("pages/3_Model_Lab.py", title="Model Lab", icon=":material/model_training:"),
    st.Page("pages/4_Fare_Estimator.py", title="Fare Estimator", icon=":material/point_of_sale:"),
    st.Page("pages/5_Insights.py", title="Insights", icon=":material/lightbulb:"),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()

with st.sidebar:
    st.divider()
    theme.side_snapshot(data_loader.load_summary(), data_loader.load_model_metrics())
