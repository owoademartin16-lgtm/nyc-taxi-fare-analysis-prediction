"""Demand & Time — when the city rides, and when the meter works hardest."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import analysis as an  # noqa: E402
from utils import charts, data_loader, theme  # noqa: E402


def peak_windows(hourly) -> None:
    table = hourly.sort_values("trips", ascending=False).head(6).reset_index(drop=True)
    total = hourly["trips"].sum()
    rows = []
    for i, row in table.iterrows():
        rows.append(
            f"<tr>"
            f"<td class='num'>{i + 1:02d}</td>"
            f"<td class='name'>{an.hour_label(int(row['pickup_hour']))}</td>"
            f"<td class='num'>{row['trips']:,}</td>"
            f"<td class='num'>{row['trips'] / total:.1%}</td>"
            f"<td class='num'>{an.money(row['revenue'], compact=True)}</td>"
            f"<td class='num'>{an.money(row['avg_total'], digits=2)}</td>"
            f"</tr>"
        )
    st.html(
        "<table class='tx-table'>"
        "<thead><tr><th>#</th><th>Hour</th><th>Trips</th><th>Share</th>"
        "<th>Revenue</th><th>Avg total</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


summary = data_loader.load_summary()
hourly = data_loader.load_table("hourly.csv")
heat = data_loader.load_table("heatmap.csv")
shifts = data_loader.load_table("shifts.csv")
dow = data_loader.load_table("dow.csv")

evening = shifts.loc[shifts["shift"] == "Evening"].iloc[0]
overnight = shifts.loc[shifts["shift"] == "Overnight"].iloc[0]

theme.header(
    "The city's <em>rhythm</em>.",
    "Demand is a clock, not a flat line. The morning builds, the evening peaks, "
    "and the weekend keeps a later schedule. This is the shape dispatch teams plan around.",
    pills=[
        f"<b>{summary['peak_hour_trips']:,}</b> trips at 6 PM",
        f"<b>{evening['share']:.1%}</b> of trips in the evening shift",
        f"<b>{summary['weekday_share']:.0%}</b> of trips Monday–Friday",
        f"<b>{an.money(summary['highest_avg_fare'], digits=2)}</b> average fare at 5 AM",
    ],
    icon="demand",
)

theme.kpis(
    [
        {
            "label": "Peak hour",
            "value": "6 PM",
            "sub": f"{summary['peak_hour_trips']:,} trips · {an.money(summary['peak_hour_revenue'], compact=True)}",
            "accent": theme.YELLOW,
        },
        {
            "label": "Busiest day",
            "value": f"Jan {summary['peak_day']}",
            "sub": f"{summary['peak_day_trips']:,} trips · {an.money(summary['peak_day_revenue'], compact=True)}",
            "accent": theme.ORANGE,
        },
        {
            "label": "Evening shift",
            "value": f"{evening['share']:.1%}",
            "sub": f"5–10 PM · {evening['trips'] / 1e6:.2f}M trips",
            "accent": theme.TEAL,
        },
        {
            "label": "Overnight shift",
            "value": f"{overnight['share']:.1%}",
            "sub": f"10 PM–6 AM · {overnight['trips'] / 1e6:.2f}M trips",
            "accent": theme.ASPHALT,
        },
        {
            "label": "Weekday trips",
            "value": f"{summary['weekday_trips'] / 1e6:.2f}M",
            "sub": f"{summary['weekday_share']:.0%} of the month",
            "accent": theme.ORANGE,
        },
        {
            "label": "Richest hour",
            "value": "5 AM",
            "sub": f"{an.money(summary['highest_avg_fare'], digits=2)} average fare",
            "accent": theme.YELLOW,
        },
    ],
    columns=6,
)

with st.container(border=True):
    st.radio(
        "Heatmap metric",
        ["Trips", "Revenue"],
        horizontal=True,
        label_visibility="collapsed",
        key="demand_heat_metric",
    )
    metric = "revenue" if st.session_state.get("demand_heat_metric") == "Revenue" else "trips"
    st.plotly_chart(
        charts.heatmap_demand(heat, metric=metric, height=390),
        config=theme.PLOTLY_CONFIG,
        use_container_width=True,
    )

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, height=386):
        st.plotly_chart(
            charts.hourly_split(hourly, height=338),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=386):
        st.plotly_chart(
            charts.avg_fare_hour(hourly, height=338),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )

left, right = st.columns([0.9, 1.1], gap="medium")
with left:
    with st.container(border=True, height=360):
        st.plotly_chart(
            charts.shift_bars(shifts, height=312),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=360):
        st.html(
            "<div style='font-family:var(--display);font-size:23px;color:var(--asphalt);"
            "letter-spacing:.03em;text-transform:uppercase;margin-bottom:2px;'>Peak windows</div>"
            "<div style='font-size:11px;color:var(--muted);margin-bottom:8px;'>"
            "The six busiest pickup hours, ranked by trip count.</div>"
        )
        peak_windows(hourly)

with st.container(border=True):
    st.plotly_chart(
        charts.dow_avg_fare(dow, height=300),
        config=theme.PLOTLY_CONFIG,
        use_container_width=True,
    )

theme.footer()
