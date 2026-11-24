"""Plotly chart builders with a single visual language."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import analysis as an
from . import theme as t

FONT = "Manrope, -apple-system, BlinkMacSystemFont, sans-serif"
DISPLAY = "Bebas Neue, Impact, sans-serif"
MONO = "IBM Plex Mono, Menlo, monospace"


# ------------------------------------------------------------------- layout ---
def base(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    entries = sum(
        1
        for trace in fig.data
        if getattr(trace, "showlegend", None) is not False and getattr(trace, "name", None)
    )
    top_margin = max(58, 18 + 21 * entries) if legend else 28
    fig.update_layout(
        height=height,
        margin=dict(l=6, r=8, t=top_margin, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=t.SUBTITLE),
        hoverlabel=dict(
            bgcolor=t.ASPHALT,
            bordercolor=t.ASPHALT,
            font=dict(family=FONT, size=11, color="#FFF8E7"),
        ),
        showlegend=legend,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color=t.MUTED),
            bgcolor="rgba(255,252,244,0.94)",
            bordercolor="rgba(216,199,169,0.9)",
            borderwidth=1,
            itemsizing="constant",
            tracegroupgap=2,
        ),
        bargap=0.38,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=t.BORDER,
        ticks="outside",
        tickcolor=t.BORDER,
        ticklen=3,
        tickfont=dict(size=10, color=t.MUTED),
        title_font=dict(size=10, color=t.MUTED),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=10, color=t.MUTED),
        title_font=dict(size=10, color=t.MUTED),
    )
    return fig


def _title(fig: go.Figure, text: str) -> None:
    fig.update_layout(
        title=dict(
            text=text,
            font=dict(family=DISPLAY, size=19, color=t.ASPHALT),
            x=0,
            xanchor="left",
            y=0.98,
            pad=dict(b=5),
        )
    )


def _empty(height: int = 300, text: str = "No data for this view") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family=FONT, size=13, color=t.MUTED),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base(fig, height=height)


# ----------------------------------------------------------------- overview ---
def hourly_demand(hourly: pd.DataFrame, height: int = 330) -> go.Figure:
    peak = hourly.loc[hourly["trips"].idxmax()]
    fig = go.Figure(
        go.Scatter(
            x=hourly["hour_label"],
            y=hourly["trips"],
            mode="lines",
            line=dict(color=t.YELLOW, width=2.6, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(245,197,24,0.18)",
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} trips<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak["hour_label"]],
            y=[peak["trips"]],
            mode="markers",
            marker=dict(color=t.ASPHALT, size=9, line=dict(color=t.YELLOW, width=2)),
            hovertemplate=f"<b>Peak · {peak['hour_label']}</b><br>{peak['trips']:,.0f} trips<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=peak["hour_label"],
        y=peak["trips"],
        text=f"<b>{peak['hour_label']}</b> · {peak['trips']:,.0f}",
        showarrow=False,
        yshift=18,
        font=dict(size=10, color=t.ASPHALT, family=MONO),
        bgcolor="rgba(255,252,244,.92)",
        bordercolor=t.BORDER,
        borderwidth=1,
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=hourly["hour_label"].tolist(),
        title_text="Pickup hour",
    )
    fig.update_yaxes(title_text="Trips", tickformat="~s")
    _title(fig, "Trips by hour · the 6 PM peak")
    return base(fig, height=height)


def hourly_revenue(hourly: pd.DataFrame, height: int = 330) -> go.Figure:
    peak_hour = int(hourly.loc[hourly["trips"].idxmax(), "pickup_hour"])
    colors = [t.YELLOW if h == peak_hour else t.ORANGE for h in hourly["pickup_hour"]]
    fig = go.Figure(
        go.Bar(
            x=hourly["hour_label"],
            y=hourly["revenue"],
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f} meter revenue<extra></extra>",
        )
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=hourly["hour_label"].tolist(),
        title_text="Pickup hour",
    )
    fig.update_yaxes(title_text="Revenue", tickprefix="$", tickformat="~s")
    _title(fig, "Meter revenue by hour")
    return base(fig, height=height)


def daily_trips(daily: pd.DataFrame, height: int = 330) -> go.Figure:
    peak = daily.loc[daily["trips"].idxmax()]
    dates = pd.to_datetime(daily["date"])
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=daily["trips"],
            mode="lines",
            line=dict(color=t.YELLOW, width=2.4, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(245,197,24,0.16)",
            hovertemplate="<b>%{x|%b %d}</b><br>%{y:,.0f} trips<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(peak["date"])],
            y=[peak["trips"]],
            mode="markers",
            marker=dict(color=t.ASPHALT, size=9, line=dict(color=t.YELLOW, width=2)),
            hovertemplate=f"<b>Jan {int(peak['pickup_day'])}</b><br>{peak['trips']:,.0f} trips<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=pd.Timestamp(peak["date"]),
        y=peak["trips"],
        text=f"<b>Jan {int(peak['pickup_day'])}</b> · {peak['trips']:,.0f}",
        showarrow=False,
        yshift=18,
        font=dict(size=10, color=t.ASPHALT, family=MONO),
        bgcolor="rgba(255,252,244,.92)",
        bordercolor=t.BORDER,
        borderwidth=1,
    )
    fig.update_xaxes(tickformat="%b %d", title_text="January 2025", nticks=7)
    fig.update_yaxes(title_text="Trips", tickformat="~s")
    _title(fig, "Daily trips · January's pulse")
    return base(fig, height=height)


def daily_revenue(daily: pd.DataFrame, height: int = 330) -> go.Figure:
    peak = daily.loc[daily["revenue"].idxmax()]
    dates = pd.to_datetime(daily["date"])
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=daily["revenue"],
            mode="lines",
            line=dict(color=t.ORANGE, width=2.4, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(217,93,57,0.13)",
            hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f} revenue<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(peak["date"])],
            y=[peak["revenue"]],
            mode="markers",
            marker=dict(color=t.ASPHALT, size=9, line=dict(color=t.ORANGE, width=2)),
            hovertemplate=f"<b>Jan {int(peak['pickup_day'])}</b><br>${peak['revenue']:,.0f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=pd.Timestamp(peak["date"]),
        y=peak["revenue"],
        text=f"<b>Jan {int(peak['pickup_day'])}</b> · ${peak['revenue'] / 1e6:.2f}M",
        showarrow=False,
        yshift=18,
        font=dict(size=10, color=t.ASPHALT, family=MONO),
        bgcolor="rgba(255,252,244,.92)",
        bordercolor=t.BORDER,
        borderwidth=1,
    )
    fig.update_xaxes(tickformat="%b %d", title_text="January 2025", nticks=7)
    fig.update_yaxes(title_text="Revenue", tickprefix="$", tickformat="~s")
    _title(fig, "Daily meter revenue")
    return base(fig, height=height)


def daytype_donut(daytype: pd.DataFrame, height: int = 330) -> go.Figure:
    weekday = daytype[daytype["pickup_day_type"] == "Weekday"].iloc[0]
    weekend = daytype[daytype["pickup_day_type"] == "Weekend"].iloc[0]
    share = weekday["trips"] / (weekday["trips"] + weekend["trips"])
    fig = go.Figure(
        go.Pie(
            labels=["Weekday", "Weekend"],
            values=[weekday["trips"], weekend["trips"]],
            hole=0.70,
            sort=False,
            direction="clockwise",
            marker=dict(
                colors=[t.YELLOW, t.ASPHALT],
                line=dict(color=t.SURFACE, width=2),
            ),
            textinfo="none",
            hovertemplate=(
                "<b>%{label}</b><br>%{value:,.0f} trips"
                "<br>%{percent} of the month<extra></extra>"
            ),
        )
    )
    fig.add_annotation(
        text=(
            f"<span style='font-family:{DISPLAY};font-size:38px;color:{t.ASPHALT};'>"
            f"{share * 100:.0f}%</span>"
            f"<br><span style='font-family:{MONO};font-size:9px;color:{t.MUTED};"
            "letter-spacing:.12em;'>WEEKDAY</span>"
        ),
        showarrow=False,
        x=0.5,
        y=0.5,
    )
    fig.add_annotation(
        text=f"{weekday['trips'] / 1e6:.2f}M vs {weekend['trips'] / 1e6:.2f}M trips",
        showarrow=False,
        x=0.5,
        y=0.28,
        font=dict(size=9.5, color=t.FAINT, family=MONO),
    )
    fig.update_layout(margin=dict(l=4, r=4, t=28, b=4), height=height)
    _title(fig, "Weekday city · weekend city")
    return base(fig, height=height)


def top_zones_chart(
    table: pd.DataFrame,
    *,
    metric: str = "trips",
    kind: str = "pickup",
    height: int = 380,
) -> go.Figure:
    if table.empty:
        return _empty(height, "No zones match this view")
    value_col = {
        "trips": f"{kind}_trips",
        "revenue": f"{kind}_revenue",
        "avg_total": f"{kind}_avg_total",
    }[metric]
    values = table[value_col]
    if metric == "revenue":
        labels = [an.money(v, compact=True) for v in values]
        tickprefix = "$"
        tickformat = "~s"
        axis_title = "Pickup revenue" if kind == "pickup" else "Drop-off revenue"
    elif metric == "avg_total":
        labels = [an.money(v, digits=2) for v in values]
        tickprefix = "$"
        tickformat = ".0f"
        axis_title = "Average total"
    else:
        labels = [f"{v:,.0f}" for v in values]
        tickprefix = ""
        tickformat = "~s"
        axis_title = "Trips"
    colors = [t.YELLOW if i < 3 else t.ORANGE if i < 6 else t.ASPHALT for i in range(len(table))]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=table["short_label"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=labels,
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=9.5, color=t.ASPHALT, family=MONO),
            customdata=table[
                [f"{kind}_trips", f"{kind}_revenue", f"{kind}_avg_total", "Borough"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]:,} trips"
                "<br>$%{customdata[1]:,.0f} revenue"
                "<br>$%{customdata[2]:,.2f} avg total"
                "<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    if metric == "avg_total":
        fig.update_xaxes(tickprefix="$", tickformat=".0f", title_text=axis_title)
    else:
        fig.update_xaxes(tickprefix=tickprefix, tickformat=tickformat, title_text=axis_title)
    span = float(values.max()) if len(values) else 1.0
    fig.update_xaxes(range=[0, span * 1.18])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, color=t.ASPHALT))
    _title(fig, f"Top {kind} zones · {metric.replace('_', ' ')}")
    fig = base(fig, height=height)
    fig.update_layout(margin=dict(l=6, r=62, t=48, b=6))
    return fig


def passenger_mix(passengers: pd.DataFrame, height: int = 380) -> go.Figure:
    table = passengers[passengers["passenger_count"].between(1, 6)].copy()
    if table.empty:
        return _empty(height)
    colors = [t.YELLOW if int(p) == 1 else t.ASPHALT for p in table["passenger_count"]]
    fig = go.Figure(
        go.Bar(
            x=[f"{int(p)}" for p in table["passenger_count"]],
            y=table["share"] * 100,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in table["share"] * 100],
            textposition="outside",
            textfont=dict(size=10, color=t.ASPHALT, family=MONO),
            customdata=table[["trips", "avg_total"]].to_numpy(),
            hovertemplate=(
                "<b>%{x} passenger(s)</b><br>%{y:.1f}% of trips"
                "<br>%{customdata[0]:,} trips"
                "<br>$%{customdata[1]:,.2f} average total<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Passengers", tickfont=dict(size=11, color=t.ASPHALT))
    fig.update_yaxes(ticksuffix="%", title_text="Share of trips", range=[0, 92])
    _title(fig, "Who rides · passenger count")
    return base(fig, height=height)


# ------------------------------------------------------------------- demand ---
def heatmap_demand(heat: pd.DataFrame, metric: str = "trips", height: int = 380) -> go.Figure:
    pivot = (
        heat.pivot_table(
            index="pickup_day_of_week",
            columns="pickup_hour",
            values=metric,
            aggfunc="first",
        )
        .reindex(an.DAY_ORDER)
        .reindex(columns=range(24))
    )
    if metric == "revenue":
        z = pivot.to_numpy()
        colorbar_title = "Revenue"
        custom = np.empty((7, 24, 2), dtype=object)
        custom[:, :, 0] = [
            [f"{int(v):,}" if pd.notna(v) else "" for v in row] for row in pivot.to_numpy()
        ]
        custom[:, :, 1] = [
            [f"${v:,.0f}" if pd.notna(v) else "" for v in row] for row in pivot.to_numpy()
        ]
        hover = "<b>%{y} · %{x}</b><br>%{customdata[1]} revenue<extra></extra>"
        template = "%{customdata[1]}"
    else:
        z = pivot.to_numpy()
        colorbar_title = "Trips"
        custom = np.empty((7, 24, 2), dtype=object)
        custom[:, :, 0] = [
            [f"{int(v):,}" if pd.notna(v) else "" for v in row] for row in pivot.to_numpy()
        ]
        custom[:, :, 1] = [
            [f"${v:,.2f}" if pd.notna(v) else "" for v in pivot.to_numpy()[i]]
            for i in range(7)
        ]
        hover = "<b>%{y} · %{x}</b><br>%{customdata[0]} trips<extra></extra>"
        template = "%{customdata[0]}"
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[an.hour_short(h) for h in range(24)],
            y=[d[:3] for d in an.DAY_ORDER],
            colorscale=t.FARE_SCALE,
            zmin=0,
            zmax=float(np.nanpercentile(z, 98)) if np.isfinite(z).any() else 1,
            xgap=2,
            ygap=2,
            showscale=True,
            colorbar=dict(
                title=dict(text=colorbar_title, font=dict(size=9, color=t.MUTED)),
                thickness=8,
                len=0.66,
                tickfont=dict(size=9, color=t.MUTED),
                outlinewidth=0,
            ),
            customdata=custom,
            hovertemplate=hover,
        )
    )
    fig.update_xaxes(title_text="Pickup hour", tickfont=dict(size=9, color=t.MUTED))
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(size=10, color=t.ASPHALT))
    _title(fig, f"Demand heatmap · {metric.replace('_', ' ')} by day and hour")
    fig = base(fig, height=height)
    fig.update_layout(margin=dict(l=6, r=8, t=46, b=34))
    return fig


def hourly_split(hourly: pd.DataFrame, height: int = 330) -> go.Figure:
    fig = go.Figure()
    for label, col, color in (
        ("Weekday", "trips_weekday", t.YELLOW),
        ("Weekend", "trips_weekend", t.TEAL),
    ):
        fig.add_trace(
            go.Scatter(
                x=hourly["hour_label"],
                y=hourly[col],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.3, shape="spline"),
                hovertemplate=f"<b>{label} · %{{x}}</b><br>%{{y:,.0f}} trips<extra></extra>",
            )
        )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=hourly["hour_label"].tolist(),
        title_text="Pickup hour",
    )
    fig.update_yaxes(title_text="Trips", tickformat="~s")
    _title(fig, "Weekday vs weekend rhythm")
    return base(fig, height=height, legend=True)


def avg_fare_hour(hourly: pd.DataFrame, height: int = 330) -> go.Figure:
    peak = hourly.loc[hourly["avg_fare"].idxmax()]
    fig = go.Figure(
        go.Scatter(
            x=hourly["hour_label"],
            y=hourly["avg_fare"],
            mode="lines+markers",
            line=dict(color=t.ORANGE, width=2.4, shape="spline"),
            marker=dict(size=4, color=t.ORANGE, line=dict(width=0)),
            fill="tozeroy",
            fillcolor="rgba(217,93,57,0.10)",
            hovertemplate="<b>%{x}</b><br>$%{y:,.2f} average fare<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak["hour_label"]],
            y=[peak["avg_fare"]],
            mode="markers",
            marker=dict(color=t.ASPHALT, size=9, line=dict(color=t.YELLOW, width=2)),
            hovertemplate=f"<b>{peak['hour_label']}</b><br>${peak['avg_fare']:,.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=peak["hour_label"],
        y=peak["avg_fare"],
        text=f"<b>{peak['hour_label']}</b> · ${peak['avg_fare']:.2f}",
        showarrow=False,
        yshift=18,
        font=dict(size=10, color=t.ASPHALT, family=MONO),
        bgcolor="rgba(255,252,244,.92)",
        bordercolor=t.BORDER,
        borderwidth=1,
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=hourly["hour_label"].tolist(),
        title_text="Pickup hour",
    )
    fig.update_yaxes(title_text="Average fare", tickprefix="$", tickformat=".0f")
    _title(fig, "Average fare by hour")
    return base(fig, height=height)


def shift_bars(shifts: pd.DataFrame, height: int = 310) -> go.Figure:
    colors = {
        "Overnight": t.ASPHALT,
        "Morning": t.YELLOW,
        "Afternoon": t.ORANGE,
        "Evening": t.TEAL,
    }
    fig = go.Figure(
        go.Bar(
            x=shifts["shift"],
            y=shifts["trips"],
            marker=dict(color=[colors.get(s, t.MUTED) for s in shifts["shift"]], line=dict(width=0)),
            text=[f"{v / 1e6:.2f}M<br>{s * 100:.0f}%" for v, s in zip(shifts["trips"], shifts["share"])],
            textposition="outside",
            textfont=dict(size=10, color=t.ASPHALT, family=MONO),
            customdata=shifts[["revenue", "avg_total"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>%{y:,.0f} trips"
                "<br>$%{customdata[0]:,.0f} revenue"
                "<br>$%{customdata[1]:,.2f} average total<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(title_text="Trips", tickformat="~s", range=[0, shifts["trips"].max() * 1.26])
    fig.update_xaxes(tickfont=dict(size=10, color=t.ASPHALT))
    _title(fig, "Trips by shift")
    return base(fig, height=height)


def dow_avg_fare(dow: pd.DataFrame, height: int = 310) -> go.Figure:
    best = dow.loc[dow["avg_fare"].idxmax(), "pickup_day_of_week"]
    colors = [t.YELLOW if d == best else t.ASPHALT for d in dow["pickup_day_of_week"]]
    fig = go.Figure(
        go.Bar(
            x=[d[:3] for d in dow["pickup_day_of_week"]],
            y=dow["avg_fare"],
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"${v:.2f}" for v in dow["avg_fare"]],
            textposition="outside",
            textfont=dict(size=9.5, color=t.ASPHALT, family=MONO),
            customdata=dow[["trips", "avg_total"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>$%{y:.2f} average fare"
                "<br>%{customdata[0]:,} trips"
                "<br>$%{customdata[1]:.2f} average total<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(title_text="Average fare", tickprefix="$", tickformat=".0f", range=[0, dow["avg_fare"].max() * 1.28])
    fig.update_xaxes(tickfont=dict(size=10, color=t.ASPHALT))
    _title(fig, "Average fare by day")
    return base(fig, height=height)


# -------------------------------------------------------------------- fares ---
def fare_distance_hex(sample: pd.DataFrame, height: int = 390) -> go.Figure:
    part = sample[(sample["trip_distance"] <= 25) & (sample["total_amount"] <= 150)]
    if part.empty:
        return _empty(height)
    fig = go.Figure(
        go.Histogram2d(
            x=part["trip_distance"],
            y=part["total_amount"],
            nbinsx=46,
            nbinsy=46,
            colorscale=t.FARE_SCALE,
            colorbar=dict(
                title=dict(text="Trips", font=dict(size=9, color=t.MUTED)),
                thickness=8,
                len=0.66,
                tickfont=dict(size=9, color=t.MUTED),
                outlinewidth=0,
            ),
            hovertemplate="Distance %{x:.1f} mi<br>Total $%{y:.0f}<br>%{z:,.0f} trips<extra></extra>",
        )
    )
    bins = np.arange(0, 25.5, 1.0)
    part = part.assign(bin=pd.cut(part["trip_distance"], bins=bins, right=False))
    median = (
        part.groupby("bin", observed=True)
        .agg(distance=("trip_distance", "median"), total=("total_amount", "median"))
        .dropna()
    )
    fig.add_trace(
        go.Scatter(
            x=median["distance"],
            y=median["total"],
            mode="lines",
            name="Median total",
            line=dict(color=t.ASPHALT, width=2.2, shape="spline"),
            hovertemplate="Distance %{x:.1f} mi<br>Median $%{y:.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Trip distance (miles)", range=[0, 25])
    fig.update_yaxes(title_text="Total amount", tickprefix="$", tickformat="~s", range=[0, 150])
    _title(fig, "Fare vs distance · model-eligible sample")
    return base(fig, height=height, legend=True)


def fare_per_mile_hour(fare_hourly: pd.DataFrame, height: int = 390) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=fare_hourly["hour_label"],
            y=fare_hourly["median_fare_per_mile"],
            marker=dict(
                color=[
                    t.YELLOW if h in (5, 6, 7, 8) else t.ORANGE
                    for h in fare_hourly["pickup_hour"]
                ],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>$%{y:.2f} median fare per mile<extra></extra>",
        )
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=fare_hourly["hour_label"].tolist(),
        title_text="Pickup hour",
    )
    fig.update_yaxes(title_text="Fare per mile", tickprefix="$", tickformat=".0f")
    _title(fig, "Median fare per mile by hour")
    return base(fig, height=height)


def zone_map(
    zones: pd.DataFrame,
    geojson: dict,
    *,
    metric: str = "trips",
    height: int = 430,
) -> go.Figure:
    table = zones[(zones["pickup_trips"] > 0) & zones["lat"].notna()].copy()
    if table.empty:
        return _empty(height)
    values, label, prefix = an.zone_metric(table, metric)
    if metric == "avg_total":
        table = table[table["pickup_trips"] >= 100].copy()
        values = table["pickup_avg_total"]
    zmax = float(np.nanpercentile(values, 96)) if len(values) else 1
    fig = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=table["LocationID"].astype(str),
            z=values,
            featureidkey="id",
            colorscale=t.FARE_SCALE,
            zmin=0,
            zmax=max(zmax, 1),
            marker=dict(line=dict(color="#FFFCF4", width=0.45), opacity=0.9),
            colorbar=dict(
                title=dict(text=label, font=dict(size=9, color=t.MUTED)),
                thickness=8,
                len=0.62,
                tickfont=dict(size=9, color=t.MUTED),
                outlinewidth=0,
            ),
            customdata=table[
                ["Zone", "Borough", "pickup_trips", "pickup_revenue", "pickup_avg_total"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}"
                "<br>%{customdata[2]:,} pickup trips"
                "<br>$%{customdata[3]:,.0f} revenue"
                "<br>$%{customdata[4]:,.2f} average total<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=40.73, lon=-73.95),
            zoom=9.25,
        )
    )
    _title(fig, f"Pickup {metric.replace('_', ' ')} by taxi zone")
    fig = base(fig, height=height)
    fig.update_layout(margin=dict(l=0, r=0, t=44, b=0))
    return fig


def airport_premium(ratecode: pd.DataFrame, height: int = 340) -> go.Figure:
    table = an.airport_rows(ratecode)
    if table.empty:
        return _empty(height)
    colors = [t.ORANGE if r in (2.0, 3.0, 4.0) else t.YELLOW for r in table["RatecodeID"]]
    fig = go.Figure(
        go.Bar(
            x=table["avg_total"],
            y=table["rate_label"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"${v:,.0f} · {s * 100:.1f}%" for v, s in zip(table["avg_total"], table["share"])],
            textposition="outside",
            textfont=dict(size=9.5, color=t.ASPHALT, family=MONO),
            customdata=table[["trips", "revenue_share"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>$%{x:,.2f} average total"
                "<br>%{customdata[0]:,} trips"
                "<br>%{customdata[1]:.1%} of revenue<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Average total", tickprefix="$", tickformat=".0f")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, color=t.ASPHALT))
    _title(fig, "Airport and special rate codes")
    fig = base(fig, height=height)
    fig.update_layout(margin=dict(l=6, r=55, t=44, b=6))
    return fig


def payment_tips(payment: pd.DataFrame, height: int = 340) -> go.Figure:
    order = ["Credit card", "Cash", "Unknown", "Dispute", "No charge"]
    table = payment[payment["payment_label"].isin(order)].copy()
    table["order"] = table["payment_label"].map({v: i for i, v in enumerate(order)})
    table = table.sort_values("order")
    colors = {
        "Credit card": t.YELLOW,
        "Cash": t.ASPHALT,
        "Unknown": t.MUTED,
        "Dispute": t.ORANGE,
        "No charge": t.FAINT,
    }
    fig = go.Figure(
        go.Bar(
            x=table["payment_label"],
            y=table["tip_rate"] * 100,
            marker=dict(color=[colors.get(v, t.MUTED) for v in table["payment_label"]], line=dict(width=0)),
            text=[f"{v * 100:.1f}%" for v in table["tip_rate"]],
            textposition="outside",
            textfont=dict(size=10, color=t.ASPHALT, family=MONO),
            customdata=table[["avg_tip", "share"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>%{y:.1f}% of total"
                "<br>$%{customdata[0]:.2f} average tip"
                "<br>%{customdata[1]:.1%} of trips<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(title_text="Tip rate", ticksuffix="%", range=[0, max(18, table["tip_rate"].max() * 115)])
    fig.update_xaxes(tickfont=dict(size=9, color=t.ASPHALT))
    _title(fig, "Tip behaviour by payment type")
    return base(fig, height=height)


def distance_bands_chart(distance_bands: pd.DataFrame, height: int = 340) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=distance_bands["band"],
            y=distance_bands["avg_total"],
            marker=dict(color=t.YELLOW, line=dict(width=0)),
            text=[f"${v:,.0f}" for v in distance_bands["avg_total"]],
            textposition="outside",
            textfont=dict(size=9.5, color=t.ASPHALT, family=MONO),
            customdata=distance_bands[["trips", "share", "median_total"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>$%{y:.2f} average total"
                "<br>%{customdata[0]:,} trips · %{customdata[1]:.1%}"
                "<br>$%{customdata[2]:.2f} median<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(title_text="Average total", tickprefix="$", tickformat=".0f")
    fig.update_xaxes(tickfont=dict(size=9, color=t.ASPHALT))
    _title(fig, "What distance does to the total")
    return base(fig, height=height)


# -------------------------------------------------------------------- model ---
def model_actual_vs_pred(sample: pd.DataFrame, metrics: dict, height: int = 390) -> go.Figure:
    part = sample[(sample["actual"] <= 120) & (sample["predicted"] <= 120)]
    if part.empty:
        return _empty(height)
    fig = go.Figure(
        go.Histogram2d(
            x=part["actual"],
            y=part["predicted"],
            nbinsx=48,
            nbinsy=48,
            colorscale=t.FARE_SCALE,
            colorbar=dict(
                title=dict(text="Trips", font=dict(size=9, color=t.MUTED)),
                thickness=8,
                len=0.66,
                tickfont=dict(size=9, color=t.MUTED),
                outlinewidth=0,
            ),
            hovertemplate="Actual $%{x:.0f}<br>Predicted $%{y:.0f}<br>%{z:,.0f} trips<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 120],
            y=[0, 120],
            mode="lines",
            name="Perfect prediction",
            line=dict(color=t.ASPHALT, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=0.72,
        y=0.18,
        xref="paper",
        yref="paper",
        text=f"R² <b>{metrics['r2']:.4f}</b>",
        showarrow=False,
        font=dict(size=12, color=t.ASPHALT, family=MONO),
        bgcolor="rgba(255,252,244,.92)",
        bordercolor=t.BORDER,
        borderwidth=1,
    )
    fig.update_xaxes(title_text="Actual total", tickprefix="$", tickformat="~s", range=[0, 120])
    fig.update_yaxes(title_text="Predicted total", tickprefix="$", tickformat="~s", range=[0, 120])
    _title(fig, "Actual vs predicted · holdout sample")
    return base(fig, height=height, legend=True)


def residual_hist(sample: pd.DataFrame, metrics: dict, height: int = 330) -> go.Figure:
    part = sample[sample["residual"].between(-15, 15)]
    fig = go.Figure(
        go.Histogram(
            x=part["residual"],
            nbinsx=54,
            marker=dict(color=t.TEAL, line=dict(width=0)),
            opacity=0.82,
            hovertemplate="Residual $%{x:.1f}<br>%{y:,.0f} trips<extra></extra>",
        )
    )
    fig.add_vline(
        x=0,
        line=dict(color=t.ASPHALT, width=1.1, dash="dash"),
        annotation_text="perfect",
        annotation_position="top",
        annotation_font=dict(size=9, color=t.MUTED, family=MONO),
    )
    fig.add_vline(
        x=metrics["residuals"]["p10"],
        line=dict(color=t.ORANGE, width=1, dash="dot"),
        annotation_text="p10",
        annotation_position="bottom",
        annotation_font=dict(size=8, color=t.ORANGE, family=MONO),
    )
    fig.add_vline(
        x=metrics["residuals"]["p90"],
        line=dict(color=t.ORANGE, width=1, dash="dot"),
        annotation_text="p90",
        annotation_position="bottom",
        annotation_font=dict(size=8, color=t.ORANGE, family=MONO),
    )
    fig.update_xaxes(title_text="Actual − predicted ($)", range=[-15, 15])
    fig.update_yaxes(title_text="Trips")
    _title(fig, "Residual distribution · holdout sample")
    return base(fig, height=height)


def model_compare(comparison: list[dict], height: int = 330) -> go.Figure:
    labels = [row["model"] for row in comparison]
    values = [row["r2"] for row in comparison]
    colors = [
        t.YELLOW if row["status"] == "Deployed" else t.TEAL if row["status"] == "Challenger" else t.SURFACE_ALT
        for row in comparison
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            textfont=dict(size=10, color=t.ASPHALT, family=MONO),
            hovertemplate="<b>%{y}</b><br>R² %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="R² on the holdout set", range=[0.80, 0.975])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, color=t.ASPHALT))
    _title(fig, "Model comparison · R²")
    fig = base(fig, height=height)
    fig.update_layout(margin=dict(l=6, r=38, t=44, b=6))
    return fig


def feature_importance(features: list[dict], height: int = 370) -> go.Figure:
    table = sorted(features, key=lambda row: row["share"], reverse=True)[:9]
    labels = [row["feature"] for row in table][::-1]
    values = [row["share"] * 100 for row in table][::-1]
    colors = [t.YELLOW if i == len(values) - 1 else t.ASPHALT for i in range(len(values))]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=9.5, color=t.ASPHALT, family=MONO),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% of model splits<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Share of splits", ticksuffix="%", range=[0, max(values) * 1.22])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, color=t.ASPHALT))
    _title(fig, "What drives the fare")
    fig = base(fig, height=height)
    fig.update_layout(margin=dict(l=6, r=38, t=44, b=6))
    return fig


def error_by_band(bands: list[dict], title: str, height: int = 320) -> go.Figure:
    labels = [row["band"] for row in bands]
    values = [row["mae"] for row in bands]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=t.ORANGE, line=dict(width=0)),
            text=[f"${v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(size=9.5, color=t.ASPHALT, family=MONO),
            customdata=[row["trips"] for row in bands],
            hovertemplate="<b>%{x}</b><br>MAE $%{y:.2f}<br>%{customdata:,} trips<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Mean absolute error", tickprefix="$", tickformat=".0f", range=[0, max(values) * 1.28])
    fig.update_xaxes(tickfont=dict(size=9, color=t.ASPHALT))
    _title(fig, title)
    return base(fig, height=height)


# ---------------------------------------------------------------- estimator ---
def comps_hist(comps: pd.DataFrame, fare: float, height: int = 250) -> go.Figure:
    if comps.empty:
        return _empty(height, "No comparable trips in the sample")
    values = comps["total_amount"].clip(lower=0, upper=200)
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=34,
            marker=dict(color=t.TEAL, line=dict(width=0)),
            opacity=0.78,
            hovertemplate="Total $%{x:.0f}<br>%{y:,.0f} similar trips<extra></extra>",
        )
    )
    fig.add_vline(
        x=fare,
        line=dict(color=t.YELLOW, width=2),
        annotation_text="model",
        annotation_position="top",
        annotation_font=dict(size=9, color=t.ASPHALT, family=MONO),
    )
    fig.update_xaxes(title_text="Observed total amount", tickprefix="$", tickformat=".0f")
    fig.update_yaxes(title_text="Trips")
    _title(fig, "Similar trips · observed totals")
    return base(fig, height=height)
