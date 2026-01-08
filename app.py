import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import gdown

# Base directory
BASE_DIR = os.path.dirname(__file__)

# CSV dataset Google Drive ID
CSV_PATH = os.path.join(BASE_DIR, "imputed_daily_AQ_2015_2025.csv")

# Download dataset if missing
if not os.path.exists(CSV_PATH):
    with st.spinner("Downloading dataset..."):
        gdown.download(f"https://drive.google.com/uc?id=1TNaUpy1iFe5EqG46_m3Sn8ZhE0gshoAk", CSV_PATH, quiet=False)

# Load model from local repo (no change to your original model logic)
model_path = os.path.join(BASE_DIR, "rf_respiratory_risk_compressed.pkl")
model = joblib.load(model_path)

# Load dataset
df = pd.read_csv(CSV_PATH)

# Feature names
FEATURE_NAMES = [
    'PM2.5_pollutant_level',
    'PM10_pollutant_level',
    'ozone_pollutant_level',
    'nitrogen_dioxide_pollutant_level',
    'carbon_monoxide_pollutant_level',
    'sulfur_dioxide_pollutant_level',
    'Light_Absorption_Coeffiecient',
    'Average_Ambient_Temperature',
    'Average_Ambient_Pressure',
    'Week_ili',
    'Year_ili'
]

# State abbreviations
STATE_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

df['date'] = pd.to_datetime(df['date'])

# Cache predictions
@st.cache_data
def add_predictions(data):
    X = data[FEATURE_NAMES]
    data['predicted_risk'] = model.predict(X)
    return data

df = add_predictions(df)
risk_labels = {0: "Low", 1: "Medium", 2: "High"}

# -------------------- Streamlit Page --------------------
st.set_page_config(page_title="Respiratory Risk Dashboard", layout="wide")
st.title("Respiratory Illness Risk Dashboard")
st.info(
    "This dashboard combines exploratory visual analysis with predictive modeling to understand how air quality, weather, "
    "and temporal factors influence respiratory illness risk."
)

tab1, tab2 = st.tabs(["Prediction & Forecasting", "Analytics & Visualization"])

# -------------------- TAB 1 --------------------
with tab1:
    st.header("Risk Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.slider("PM2.5", 0.0, 300.0, 50.0)
        pm10 = st.slider("PM10", 0.0, 300.0, 80.0)
        ozone = st.slider("Ozone", 0.0, 0.2, 0.05)
        no2 = st.slider("NO₂", 0.0, 0.2, 0.05)

    with col2:
        co = st.slider("CO", 0.0, 5.0, 1.0)
        so2 = st.slider("SO₂", 0.0, 0.2, 0.02)
        lac = st.slider("Light Absorption Coefficient", 0.0, 1.0, 0.2)

    with col3:
        temp = st.slider("Temperature (°C)", -10.0, 45.0, 25.0)
        pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, 1013.0)
        week = st.slider("Epidemiological Week", 1, 52, 20)
        year = st.slider("Year", 2015, 2025, 2022)

    input_df = pd.DataFrame([[
        pm25, pm10, ozone, no2, co, so2, lac,
        temp, pressure, week, year
    ]], columns=FEATURE_NAMES)

    current_pred = model.predict(input_df)[0]
    next_week_df = input_df.copy()
    next_week_df['Week_ili'] = min(52, week + 1)
    next_week_pred = model.predict(next_week_df)[0]

    st.subheader("Prediction Results")
    st.success(f"**Current Risk:** {risk_labels[current_pred]}")
    st.info(f"**Next Week Forecast:** {risk_labels[next_week_pred]}")

    # Risk trend over time
    st.subheader("Risk Trend Over Time")
    state_for_trend = st.selectbox("Select State (optional)", ["All"] + sorted(df['state_name'].unique()))
    trend_df = df.copy()
    if state_for_trend != "All":
        trend_df = trend_df[trend_df['state_name'] == state_for_trend]
    weekly_trend = trend_df.groupby(['Year_ili', 'Week_ili'])['predicted_risk'].mean().reset_index()
    fig = px.line(
        weekly_trend,
        x="Week_ili",
        y="predicted_risk",
        color="Year_ili",
        labels={"predicted_risk": "Mean Risk Level"},
        title="Seasonal Risk Pattern"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 2 --------------------
with tab2:
    st.header("Risk Analytics")

    col1, col2, col3 = st.columns(3)
    with col1:
        year_range = st.slider("Year Range", 2015, 2025, (2018, 2022))
        agg_method = st.radio("Aggregation Method", ["Mean", "Median"], horizontal=True)
    with col2:
        selected_states = st.multiselect("Select States", sorted(df['state_name'].unique()), default=[])
    with col3:
        risk_filter = st.multiselect("Risk Levels", ["Low", "Medium", "High"], default=["High"])

    filtered = df[
        (df['Year_ili'] >= year_range[0]) &
        (df['Year_ili'] <= year_range[1]) &
        (df['predicted_risk'].map(risk_labels).isin(risk_filter))
    ]
    if selected_states:
        filtered = filtered[filtered['state_name'].isin(selected_states)]

    risk_agg = 'mean' if agg_method == "Mean" else 'median'
    state_summary = filtered.groupby(['Year_ili', 'state_name']).agg(
        mean_risk=('predicted_risk', risk_agg),
        avg_pm25=('PM2.5_pollutant_level', 'mean'),
        count=('predicted_risk', 'count')
    ).reset_index()
    state_summary['state_code'] = state_summary['state_name'].map(STATE_ABBR)
    state_summary = state_summary.dropna(subset=['state_code'])
    state_summary['risk_label'] = state_summary['mean_risk'].round().map({0: 'Low', 1: 'Medium', 2: 'High'})

    # Map
    fig_map = px.choropleth(
        state_summary,
        locations='state_code',
        locationmode='USA-states',
        color='risk_label',
        scope='usa',
        animation_frame='Year_ili',
        hover_name='state_name',
        hover_data={'mean_risk': ':.2f', 'avg_pm25': ':.1f', 'count': True, 'state_code': False},
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
        title=f"Respiratory Risk by State ({agg_method} Aggregation)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Feature importance
    st.subheader("Pollutant Contribution Analysis")
    importances = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    fig_imp = px.bar(importances, x="Importance", y="Feature", orientation='h', title="Random Forest Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Temporal Trends
    st.header("Temporal Trends in Respiratory Illness Risk")
    state_sel = st.selectbox("Select State", sorted(df["state_name"].unique()))
    year_range = st.slider("Select Year Range", int(df["Year_ili"].min()), int(df["Year_ili"].max()), (2015, 2025))
    temp_df = df[(df["state_name"] == state_sel) & (df["Year_ili"].between(year_range[0], year_range[1]))]
    yearly_trend = temp_df.groupby("Year_ili")["ILI_Target"].mean().reset_index()
    fig_trend = px.line(yearly_trend, x="Year_ili", y="ILI_Target", markers=True, title=f"Average Respiratory Illness Risk Over Time ({state_sel})", labels={"Year_ili": "Year", "ILI_Target": "Average ILI (%)"})
    st.plotly_chart(fig_trend, use_container_width=True)

    # Seasonal patterns
    st.header("Seasonal Patterns of Respiratory Illness")
    season_df = temp_df.copy()
    season_df["month"] = season_df["date"].dt.month
    fig_season = px.box(season_df, x="month", y="ILI_Target", title="Monthly Distribution of Respiratory Illness Risk", labels={"month": "Month", "ILI_Target": "ILI (%)"})
    st.plotly_chart(fig_season, use_container_width=True)

    # Pollution vs Risk
    st.header("Pollution vs Respiratory Risk")
    pollutant = st.selectbox("Select Pollutant", ["PM2.5_pollutant_level", "PM10_pollutant_level", "ozone_pollutant_level", "nitrogen_dioxide_pollutant_level"])
    fig_scatter = px.scatter(temp_df, x=pollutant, y="ILI_Target", opacity=0.6, trendline="ols", title=f"{pollutant.replace('_', ' ')} vs Respiratory Illness Risk", labels={pollutant: "Pollutant Level", "ILI_Target": "ILI (%)"})
    st.plotly_chart(fig_scatter, use_container_width=True)
