# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -------------------------
# Paths
# -------------------------
MODELS_DIR = Path("Models")
RESULTS_DIR = Path("Model Results")

# Load results
results_files = list(RESULTS_DIR.glob("model_results_*.xlsx"))
metrics_df = pd.read_excel(results_files[-1]) if results_files else pd.DataFrame()

# Load scaler
scaler_path = MODELS_DIR / "scaler.pkl"
if not scaler_path.exists():
    st.error(" scaler.pkl not found! Please run main.py first.")
    st.stop()
else:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

# Models available
MODEL_PATHS = {
    "LogisticRegression": MODELS_DIR / "LogisticRegression_diabetes.pkl",
    "SVM": MODELS_DIR / "SVM_diabetes.pkl",
    "RandomForest": MODELS_DIR / "RandomForest_diabetes.pkl",
    "XGBoost": MODELS_DIR / "XGBoost_diabetes.pkl",
}

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="🩺 Diabetes Prediction", layout="wide")
st.title("🩺 CDC Diabetes Health Indicators Prediction System")

# -------------------------
# Model selection
# -------------------------
model_choice = st.selectbox("Choose a trained model:", list(MODEL_PATHS.keys()))

# Show metrics
if not metrics_df.empty:
    st.write("### Model Metrics")
    model_metrics = metrics_df[metrics_df["Model"] == model_choice]
    st.dataframe(model_metrics)

# Load model
if not MODEL_PATHS[model_choice].exists():
    st.error(f" {MODEL_PATHS[model_choice].name} not found! Please run main.py first.")
    st.stop()
else:
    with open(MODEL_PATHS[model_choice], "rb") as f:
        model = pickle.load(f)

# -------------------------
# Single Input Prediction
# -------------------------
st.subheader("🔹 Single Patient Prediction")

feature_names = [
    "HighBP (High Blood Pressure)",
    "HighChol (High Cholesterol)",
    "CholCheck (Cholesterol Check in 5 years)",
    "BMI (Body Mass Index)",
    "Smoker (Ever Smoked 100 Cigarettes)",
    "Stroke (History of Stroke)",
    "HeartDiseaseorAttack (Heart Disease or MI)",
    "PhysActivity (Physical Activity in last 30 days)"
]

inputs = {}
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        raw_name = feature.split()[0]
        if raw_name == "BMI":
            val = st.number_input(feature, min_value=10, max_value=60, value=25, key=f"feat_{i}")
        else:
            val = st.selectbox(feature, [0, 1], key=f"feat_{i}")
        inputs[raw_name] = val

if st.button("Predict Diabetes"):
    single_df = pd.DataFrame([inputs])
    single_scaled = scaler.transform(single_df)

    pred = model.predict(single_scaled)[0]
    prob = model.predict_proba(single_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Prediction: **At Risk of Diabetes** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Prediction: **Healthy (No Diabetes)** (Probability: {prob:.2f})")

# -------------------------
# Batch Prediction
# -------------------------
st.subheader("📂 Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV with features", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(user_df.head())

    expected_cols = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity"]
    if not all(col in user_df.columns for col in expected_cols):
        st.error("CSV must contain required columns: " + ", ".join(expected_cols))
    else:
        scaled = scaler.transform(user_df[expected_cols])
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:,1]

        user_df["Prediction"] = ["At Risk" if p == 1 else "Healthy" for p in preds]
        user_df["Diabetes_Probability"] = probs

        st.write("### Predictions")
        st.dataframe(user_df.head())

        csv = user_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="diabetes_predictions.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>Created by: Samruddhi R. Panhalkar</strong></p>
        <p><strong>Roll No: USN- 2MM22RI014</strong></p>
        <p>📧 samruddhipanhalkar156@gmail.com | 📱 +91-8951831491</p>
        <p>🏫 Maratha Mandal Engineering College</p>
    </div>
    """,
    unsafe_allow_html=True,
)

