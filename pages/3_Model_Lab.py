"""Model Lab — holdout performance, drivers and error profile."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import analysis as an  # noqa: E402
from utils import charts, data_loader, theme  # noqa: E402

metrics = data_loader.load_model_metrics()
sample = data_loader.load_model_sample()
m = metrics["metrics"]
r = metrics["residuals"]
split = metrics["split"]
card = metrics["model_card"]

theme.header(
    "The <em>fare engine</em>.",
    "A LightGBM regressor predicts the total amount from trip context: distance, "
    "zones, time, rate code and payment type. It never sees the fare components it is "
    "trying to explain.",
    pills=[
        f"<b>{an.money(m['mae'], digits=2)}</b> MAE",
        f"<b>{m['r2']:.4f}</b> holdout R²",
        f"<b>{r['within_5']:.1%}</b> of errors within $5",
        f"<b>{split['test']:,}</b> holdout trips",
    ],
    icon="model",
)

theme.kpis(
    [
        {
            "label": "MAE",
            "value": an.money(m["mae"], digits=2),
            "sub": "average absolute error",
            "accent": theme.YELLOW,
        },
        {
            "label": "RMSE",
            "value": an.money(m["rmse"], digits=2),
            "sub": "penalises large misses",
            "accent": theme.ORANGE,
        },
        {
            "label": "Holdout R²",
            "value": f"{m['r2']:.4f}",
            "sub": "variance explained",
            "accent": theme.TEAL,
        },
        {
            "label": "Train R²",
            "value": f"{m['train_r2']:.4f}",
            "sub": f"{split['train']:,} training trips",
            "accent": theme.ASPHALT,
        },
        {
            "label": "Train–test gap",
            "value": f"{m['gap']:.4f}",
            "sub": "no meaningful overfit",
            "accent": theme.YELLOW,
        },
        {
            "label": "Within $5",
            "value": f"{r['within_5']:.1%}",
            "sub": f"{r['within_2']:.1%} within $2",
            "accent": theme.ORANGE,
        },
    ],
    columns=6,
)

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, height=440):
        st.plotly_chart(
            charts.model_actual_vs_pred(sample, m, height=390),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=440):
        st.plotly_chart(
            charts.residual_hist(sample, metrics, height=390),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, height=420):
        st.plotly_chart(
            charts.model_compare(metrics["comparison"], height=370),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=420):
        st.plotly_chart(
            charts.feature_importance(metrics["feature_importance"], height=370),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )

left, right = st.columns(2, gap="medium")
with left:
    with st.container(border=True, height=372):
        st.plotly_chart(
            charts.error_by_band(metrics["fare_bands"], "MAE by actual fare band", height=322),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )
with right:
    with st.container(border=True, height=372):
        st.plotly_chart(
            charts.error_by_band(
                metrics["distance_bands"], "MAE by trip distance", height=322
            ),
            config=theme.PLOTLY_CONFIG,
            use_container_width=True,
        )

with st.container(border=True):
    st.html(
        "<div class='tx-card' style='padding:0;border:0;background:transparent;'>"
        "<div class='tx-kicker' style='margin-bottom:2px;'>MODEL CARD</div>"
        "<h4>LightGBM · total amount</h4>"
        "<p style='max-width:880px;'>A gradient-boosted tree model trained on a random "
        "80/20 split of the model-eligible January trips. The target is the total amount "
        "paid, including tips, tolls and fees.</p>"
        "</div>"
    )
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.metric("Training rows", f"{split['train']:,}")
        st.metric("Holdout rows", f"{split['test']:,}")
    with c2:
        st.metric("Trees", f"{card['n_estimators']}")
        st.metric("Learning rate", f"{card['learning_rate']}")
    with c3:
        st.metric("Leaves", f"{card['num_leaves']}")
        st.metric("Random seed", f"{card['random_state']}")
    with c4:
        st.metric("Features", f"{card['feature_count']}")
        st.metric("Target", "total amount")
    st.html(
        "<div style='margin-top:4px;'>"
        + theme.chips(
            [
                ("TRIP DISTANCE", "yellow"),
                ("PICKUP ZONE", "orange"),
                ("DROP-OFF ZONE", "orange"),
                ("RATE CODE", "teal"),
                ("PICKUP HOUR", "info"),
                ("PAYMENT TYPE", "dark"),
                ("DAY TYPE", "info"),
                ("PASSENGERS", "teal"),
            ]
        )
        + "</div>"
    )
    with st.expander("Method notes"):
        st.markdown(
            "- The train/test split is the case-study split: 80/20, `random_state=42`.\n"
            "- The preprocessor is fitted on the training set only; the model never sees "
            "the holdout during fitting.\n"
            "- Features are raw trip context. `fare_amount`, `tip_amount`, `tolls_amount` "
            "and other fare components are excluded, so the model cannot leak the answer.\n"
            "- Feature importance is measured by the share of tree splits, not by "
            "predictive gain. Distance is the strongest single driver; zones and rate "
            "code carry the location and product premium.\n"
            "- The portable scikit-learn fallback scores "
            f"R² {metrics['fallback']['r2']:.4f}, MAE "
            f"{an.money(metrics['fallback']['mae'], digits=2)}, and is only used on "
            "local machines where LightGBM cannot load."
        )

theme.footer()
