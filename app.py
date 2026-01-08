import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import gdown

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "rf_respiratory_risk_compressed.pkl")
CSV_PATH = os.path.join(BASE_DIR, "imputed_daily_AQ_2015_2025.csv")

# ------------------------
# Download dataset if missing
# ------------------------
if not os.path.exists(CSV_PATH):
    with st.spinner("Downloading dataset from Google Drive..."):
        # Replace YOUR_FILE_ID with actual Google Drive ID
        gdown.download(
            f"https://drive.google.com/uc?id=1TNaUpy1iFe5EqG46_m3Sn8ZhE0gshoAk",
            CSV_PATH,
            quiet=False
        )

# ------------------------
# Load model
# ------------------------
model = joblib.load(MODEL_PATH)

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])

# ------------------------
# Features and state codes
# ------------------------
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

# ------------------------
# Cache predictions
# ------------------------
@st.cache_data
def add_predictions(data):
    X = data[FEATURE_NAMES]
    data['predicted_risk'] = model.predict(X)
    return data

df = add_predictions(df)
risk_labels = {0: "Low", 1: "Medium", 2: "High"}

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Respiratory Risk Dashboard", layout="wide")
st.title("Respiratory Illness Risk Dashboard")
st.info(
    "This dashboard combines exploratory visual analysis with predictive modeling to understand how air quality, weather, "
    "and temporal factors influence respiratory illness risk."
)

tab1, tab2 = st.tabs(["Prediction & Forecasting", "Analytics & Visualization"])

# ------------------------
# TAB 1: Prediction & Forecasting
# ------------------------
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

# ------------------------
# TAB 2: Analytics & Visualization
# ------------------------
with tab2:
    st.header("Respiratory Risk Analytics")
    st.write("Filters and visualization logic here...")  
    # You can keep all the analytics/plotly visualizations from your original code
    # Make sure they use df with cached predictions
