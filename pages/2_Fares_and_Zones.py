"""Fares & Zones — what the meter charges, and where the city pays it."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import analysis as an  # noqa: E402
from utils import charts, data_loader, theme  # noqa: E402


def zone_table(table: pd.DataFrame, kind: str) -> None:
    prefix = "pickup" if kind == "pickup" else "dropoff"
    rows = []
    for _, row in table.iterrows():
        rows.append(
            f"<tr>"
            f"<td class='name'>{row['short_label']}</td>"
            f"<td class='subtle'>{row['Borough']}</td>"
            f"<td class='num'>{int(row[f'{prefix}_trips']):,}</td>"
            f"<td class='num'>{an.money(row[f'{prefix}_revenue'], compact=True)}</td>"
            f"<td class='num'>{an.money(row[f'{prefix}_avg_total'], digits=2)}</td>"
            f"</tr>"
        )
    st.html(
        "<table class='tx-table'>"
        "<thead><tr><th>Zone</th><th>Borough</th><th>Trips</th>"
        "<th>Revenue</th><th>Avg total</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


summary = data_loader.load_summary()
sample = data_loader.load_sample()
zones = data_loader.load_zones()
geojson = data_loader.load_zone_geojson()
ratecode = data_loader.load_table("ratecode.csv")
payment = data_loader.load_table("payment.csv")
distance_bands = data_loader.load_table("distance_bands.csv")
fare_hourly = data_loader.load_table("fare_hourly.csv")

median_fare_per_mile = float(sample["fare_per_mile"].replace([float("inf"), float("-inf")], pd.NA).dropna().median())

theme.header(
    "Follow the <em>meter</em>.",
    "A fare is a distance signal first, then a story about where the trip starts, "
    "when it happens, and which rate code applies. The map shows where the city pays.",
    pills=[
        f"<b>{an.money(summary['avg_total'], digits=2)}</b> average total",
        f"<b>{summary['median_distance']:.1f} mi</b> median trip",
        f"<b>{an.money(median_fare_per_mile, digits=2)}</b> median fare per mile",
        f"<b>{summary['airport_share']:.2%}</b> airport trips · {summary['airport_revenue_share']:.1%} of revenue",
    ],
    icon="fares",
)

theme.kpis(
    [
        {
            "label": "Average total",
            "value": an.money(summary["avg_total"], digits=2),
            "sub": f"{an.money(summary['median_total'], digits=2)} median",
            "accent": theme.YELLOW,
        },
        {
            "label": "Median trip",
            "value": f"{summary['median_distance']:.1f} mi",
            "sub": "half of trips are shorter",
            "accent": theme.ASPHALT,
        },
        {
            "label": "Median fare / mile",
            "value": an.money(median_fare_per_mile, digits=2),
            "sub": "model-eligible sample",
            "accent": theme.ORANGE,
        },
        {
            "label": "Card tip rate",
            "value": f"{summary['card_tip_rate']:.1%}",
            "sub": "tips as a share of card totals",
            "accent": theme.TEAL,
        },
        {
            "label": "JFK + Newark",
            "value": f"{summary['airport_trips']:,}",
            "sub": f"{summary['airport_share']:.2%} of trips",
            "accent": theme.ORANGE,
        },
        {
            "label": "Airport revenue",
            "value": an.money(summary["airport_revenue"], compact=True),
            "sub": f"{summary['airport_revenue_share']:.1%} of all meter revenue",
            "accent": theme.YELLOW,
        },
    ],
    columns=6,
)

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, height=442):
        st.plotly_chart(
            charts.fare_distance_hex(sample, height=392),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=442):
        st.plotly_chart(
            charts.fare_per_mile_hour(fare_hourly, height=392),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )

left, right = st.columns([1.08, 0.92], gap="medium")
with left:
    with st.container(border=True):
        st.radio(
            "Map metric",
            ["Pickup trips", "Pickup revenue", "Average total"],
            horizontal=True,
            label_visibility="collapsed",
            key="zone_map_metric",
        )
        metric = {
            "Pickup trips": "trips",
            "Pickup revenue": "revenue",
            "Average total": "avg_total",
        }[st.session_state.get("zone_map_metric", "Pickup trips")]
        st.plotly_chart(
            charts.zone_map(zones, geojson, metric=metric, height=418),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True):
        st.radio(
            "Zone ranking",
            ["Pickup zones", "Drop-off zones"],
            horizontal=True,
            label_visibility="collapsed",
            key="zone_rank_kind",
        )
        kind = "pickup" if st.session_state.get("zone_rank_kind") == "Pickup zones" else "dropoff"
        table = an.top_zones(zones, kind=kind, metric="trips", n=10)
        st.plotly_chart(
            charts.top_zones_chart(table, metric="trips", kind=kind, height=418),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, height=388):
        st.plotly_chart(
            charts.airport_premium(ratecode, height=340),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=388):
        st.plotly_chart(
            charts.payment_tips(payment, height=340),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )

left, right = st.columns([0.46, 0.54], gap="medium")
with left:
    with st.container(border=True, height=402):
        st.plotly_chart(
            charts.distance_bands_chart(distance_bands, height=352),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=402):
        st.html(
            "<div style='font-family:var(--display);font-size:23px;color:var(--asphalt);"
            "letter-spacing:.03em;text-transform:uppercase;margin-bottom:2px;'>Top pickup zones</div>"
            "<div style='font-size:11px;color:var(--muted);margin-bottom:8px;'>"
            "Ranked by January pickup trips. Revenue is meter revenue.</div>"
        )
        table = an.top_zones(zones, kind="pickup", metric="trips", n=10)
        zone_table(table, "pickup")
        download = an.top_zones(zones, kind="pickup", metric="trips", n=25)[
            ["LocationID", "Zone", "Borough", "pickup_trips", "pickup_revenue", "pickup_avg_total"]
        ].to_csv(index=False)
        st.download_button(
            "Download top 25 zones",
            data=download,
            file_name="nyc_taxi_top_pickup_zones.csv",
            mime="text/csv",
            use_container_width=True,
        )

theme.footer()
