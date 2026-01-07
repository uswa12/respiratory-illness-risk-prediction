Respiratory Illness Risk Prediction 


Project Overview :

This project predicts respiratory illness risk using a combination of air pollution, weather, and seasonal epidemiological features, it leverages machine learning models to estimate whether environmental conditions pose a Low, Medium, or High risk to respiratory health and provides an interactive dashboard for prediction, forecasting, and spatial analysis.
The system is designed to support public health awareness, early warning, and data driven insights into smog related respiratory risks.


Objectives :

- Predict respiratory illness risk based on smog and weather data

- Compare multiple machine learning models

- Select and deploy the best performing model

- Visualize risk trends across time and US states

- Provide short term forecasting insights


Dataset Description :

- Records: ~213,000 rows

- Time span: 2015 – 2025

- Geographic scope: United States (state level)

- Key Features Used

- Air Quality Indicators

State names are retained for analysis and visualization, but not used as model inputs to improve generalization.



Project Structure :

Respiratory-illness-risk-predictor/


── app.py                     

── rf_respiratory_risk.pkl   

── scaler.pkl

── imputed_daily_AQ_2015_2025.csv

── model.ipynb                

── requirements.txt

── venv/


Installation & Setup

A) Create and activate virtual environment

python3 -m venv venv

source venv/bin/activate


B) Install dependencies

pip install -r requirements.txt


C)  the Streamlit app

streamlit run app.py


The app will be available at:
http://localhost:8501


Dependencies :

Python 3.9+

streamlit

pandas

numpy

scikit-learn

joblib

plotly

matplotlib

seaborn

statsmodels

