import joblib

model = joblib.load("rf_respiratory_risk.pkl")
joblib.dump(model, "rf_respiratory_risk_compressed.pkl", compress=3)
print("Compressed model saved!")

