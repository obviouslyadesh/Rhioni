"""
Real-Time Accident Severity Prediction
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────
# PATHS  (adjust ARTIFACT_DIR to where your .joblib files live)
# ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artifacts")


# ──────────────────────────────────────────────────────────────────
# LOAD MODEL & METADATA  (cached so we only load once)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
    model        = joblib.load(os.path.join(ARTIFACT_DIR, "lgbm_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    return preprocessor, model, metadata

preprocessor, model, meta = load_artifacts()


# ──────────────────────────────────────────────────────────────────
# HELPER: FEATURE ENGINEERING (mirrors notebook)
# ──────────────────────────────────────────────────────────────────
def build_input_row(hour, minute, day_of_week, light_condition,
                    road_geometry, speed_zone, highway):
    """Replicates the notebook's engineer_features() for a single row."""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    bins   = [-1, 5, 11, 16, 20, 23]
    labels = ["NIGHT", "MORNING", "AFTERNOON", "EVENING", "LATE_NIGHT"]
    tod_idx    = int(pd.cut([hour], bins=bins, labels=False)[0])
    time_of_day = labels[tod_idx]

    if speed_zone <= 50:
        speed_risk = "LOW"
    elif speed_zone <= 70:
        speed_risk = "MEDIUM"
    elif speed_zone <= 90:
        speed_risk = "HIGH"
    else:
        speed_risk = "VERY_HIGH"

    is_peak    = int(hour in [7, 8, 9, 17, 18, 19])
    is_weekend = int(day_of_week in [6, 7])

    return pd.DataFrame([{
        "HOUR": hour, "MINUTE": minute,
        "HOUR_SIN": hour_sin, "HOUR_COS": hour_cos,
        "DAY_OF_WEEK": day_of_week, "LIGHT_CONDITION": light_condition,
        "SPEED_ZONE": speed_zone, "IS_PEAK_HOUR": is_peak, "IS_WEEKEND": is_weekend,
        "ROAD_GEOMETRY_DESC": road_geometry, "HIGHWAY": highway,
        "TIME_OF_DAY": time_of_day, "SPEED_RISK": speed_risk
    }])


def predict(row_df):
    X_proc = preprocessor.transform(row_df)
    pred   = model.predict(X_proc)[0]
    proba  = model.predict_proba(X_proc)[0]
    return int(pred) + 1, proba          # severity 1-4, raw probabilities


# ──────────────────────────────────────────────────────────────────
# SEVERITY STYLING
# ──────────────────────────────────────────────────────────────────
SEVERITY_META = {
    1: {"label": "Fatal",         "color": "#d62728", "emoji": "💀"},
    2: {"label": "Serious Injury","color": "#ff7f0e", "emoji": "🏥"},
    3: {"label": "Other Injury",  "color": "#ffdd57", "emoji": "⚠️"},
    4: {"label": "Non-Injury",    "color": "#2ca02c", "emoji": "✅"},
}


# ──────────────────────────────────────────────────────────────────
# SIDEBAR – ABOUT
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 About")
    st.info(
        "This tool predicts **accident severity** *before* an accident happens, "
        "using only pre-accident contextual features."
    )
    st.markdown("### Model Metrics")
    metrics = meta.get("model_metrics", {})
    col1, col2 = st.columns(2)
    col1.metric("Accuracy",    f"{metrics.get('accuracy', 0)*100:.1f}%")
    col2.metric("Macro F1",    f"{metrics.get('macro_f1', 0)*100:.1f}%")
    col1.metric("Weighted F1", f"{metrics.get('weighted_f1', 0)*100:.1f}%")
    col2.metric("Kappa",       f"{metrics.get('cohen_kappa', 0):.3f}")

    st.markdown("---")
    st.caption("Model: LightGBM · Features: 13 · Classes: 4")


# ──────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────
st.title("🚗 Real-Time Accident Severity Predictor")
st.markdown(
    "Fill in the **road & time conditions** below and click **Predict** "
    "to get the severity class with probabilities."
)
st.markdown("---")


# ──────────────────────────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("🕐 Time & Day")
    c1, c2, c3 = st.columns(3)

    now = datetime.now()
    hour        = c1.slider("Hour of day", 0, 23, now.hour)
    minute      = c2.slider("Minute",      0, 59, now.minute)
    day_of_week = c3.selectbox(
        "Day of Week",
        options=[1,2,3,4,5,6,7],
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1],
        index=now.weekday()
    )

    st.subheader("🛣️ Road Conditions")
    c4, c5, c6 = st.columns(3)

    light_label_map = meta["light_conditions"]
    light_options   = list(light_label_map.keys())
    light_condition = c4.selectbox(
        "Light Condition",
        options=[int(k) for k in light_options],
        format_func=lambda x: light_label_map[str(x)]
    )

    speed_zone = c5.selectbox("Speed Zone (km/h)", meta["speed_zones"], index=4)

    road_geometry = c6.selectbox("Road Geometry", meta["road_geometry_options"])

    highway = st.selectbox("Highway Segment", meta["highway_options"])

    submitted = st.form_submit_button("🔍 Predict Severity", use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# PREDICTION & DISPLAY
# ──────────────────────────────────────────────────────────────────
if submitted:
    with st.spinner("Running model…"):
        time.sleep(0.3)   # UX: tiny delay so spinner shows
        row      = build_input_row(hour, minute, day_of_week, light_condition,
                                   road_geometry, speed_zone, highway)
        sev, proba = predict(row)

    info = SEVERITY_META[sev]
    st.markdown("---")
    st.subheader("Prediction Result")

    # ── Banner ────────────────────────────────────────────────────
    banner_css = f"background:{info['color']};padding:18px 24px;border-radius:8px;color:white;"
    st.markdown(
        f"<div style='{banner_css}'>"
        f"<h2 style='margin:0'>{info['emoji']} Predicted Severity: {sev} — {info['label']}</h2>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(" ")
    col_a, col_b = st.columns([1, 1])

    # ── Probability bar chart ─────────────────────────────────────
    with col_a:
        st.markdown("#### Class Probabilities")
        labels = [f"Sev {i+1} – {SEVERITY_META[i+1]['label']}" for i in range(4)]
        colors = [SEVERITY_META[i+1]['color'] for i in range(4)]
        fig = go.Figure(go.Bar(
            x=[f"{p*100:.1f}%" for p in proba],
            y=labels,
            orientation='h',
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in proba],
            textposition='outside'
        ))
        fig.update_layout(
            height=260, margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Input Summary ─────────────────────────────────────────────
    with col_b:
        st.markdown("#### Input Summary")
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        light_names = meta["light_conditions"]
        st.table(pd.DataFrame({
            "Feature": [
                "Time", "Day", "Light Condition",
                "Speed Zone", "Road Geometry", "Highway"
            ],
            "Value": [
                f"{hour:02d}:{minute:02d}",
                day_names[day_of_week - 1],
                light_names[str(light_condition)],
                f"{speed_zone} km/h",
                road_geometry,
                highway
            ]
        }))

    # ── Risk insight ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Risk Insight")
    risk_map = {
        1: "⛔ **Fatal risk** — extreme caution. Conditions are highly dangerous.",
        2: "🚨 **Serious injury risk** — drive carefully and maintain safety margins.",
        3: "⚠️ **Moderate injury risk** — stay alert and observe speed limits.",
        4: "✅ **Low risk** — conditions appear safe. Continue with normal caution.",
    }
    st.info(risk_map[sev])


# ──────────────────────────────────────────────────────────────────
# BATCH PREDICTION TAB
# ──────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Batch Prediction (Upload CSV)"):
    st.markdown(
        "Upload a CSV with columns: `ACCIDENT_TIME`, `DAY_OF_WEEK`, "
        "`LIGHT_CONDITION`, `ROAD_GEOMETRY_DESC`, `SPEED_ZONE`, `HIGHWAY`"
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df):,} rows")

        try:
            from src.predict import batch_predict   # optional module
            results = batch_predict(batch_df, preprocessor, model)
        except ImportError:
            # Inline batch logic
            def engineer_batch(df):
                df = df.copy()
                df['ACCIDENT_TIME'] = pd.to_datetime(df['ACCIDENT_TIME'], format='%H:%M:%S', errors='coerce')
                df['HOUR']   = df['ACCIDENT_TIME'].dt.hour.fillna(0).astype(int)
                df['MINUTE'] = df['ACCIDENT_TIME'].dt.minute.fillna(0).astype(int)
                df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
                df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
                bins   = [-1, 5, 11, 16, 20, 23]
                labels = ['NIGHT','MORNING','AFTERNOON','EVENING','LATE_NIGHT']
                df['TIME_OF_DAY'] = pd.cut(df['HOUR'], bins=bins, labels=labels).astype(str)
                df['IS_PEAK_HOUR'] = df['HOUR'].isin([7,8,9,17,18,19]).astype(int)
                df['IS_WEEKEND']   = df['DAY_OF_WEEK'].isin([6,7]).astype(int)
                df['SPEED_RISK'] = pd.cut(
                    df['SPEED_ZONE'], bins=[0,50,70,90,999],
                    labels=['LOW','MEDIUM','HIGH','VERY_HIGH']
                ).astype(str)
                return df

            FEATS = meta['all_features']
            batch_eng = engineer_batch(batch_df)
            X_batch   = preprocessor.transform(batch_eng[FEATS])
            preds     = model.predict(X_batch) + 1
            probas    = model.predict_proba(X_batch)
            batch_df['PREDICTED_SEVERITY'] = preds
            for i in range(4):
                batch_df[f'PROB_SEV_{i+1}'] = (probas[:, i] * 100).round(2)
            results = batch_df

        st.dataframe(results.head(50))
        csv_out = results.to_csv(index=False).encode()
        st.download_button("⬇️ Download Predictions", csv_out, "predictions.csv", "text/csv")
