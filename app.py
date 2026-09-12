import streamlit as st
st.title("تطبيق التنبؤ بحدوث الحرائق")
st.write("التطبيق يعمل ويتم تحميل النموذج الآن...")


# ============================================================
# FOREST FIRE RISK PREDICTION
# Streamlit Application
# ============================================================

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# LOAD FINAL MODELS
# ============================================================

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Path to the model file
MODEL_PATH = BASE_DIR / "forest_fire_final_model.pkl"

# Check if the model file exists
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}"
    )
    st.stop()

# Load the model configuration
model_config = joblib.load(MODEL_PATH)


# ============================================================
# EXTRACT MODELS AND SETTINGS
# ============================================================

# Final XGBoost model
model = model_config["model"]

# Decision Tree model
dt_model = model_config["dt_model"]

# SVM model
svm_model = model_config["svm_model"]

# Top-10 features used by all models
features = model_config["features"]

# Final Fire / No Fire decision threshold
threshold = model_config["threshold"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Forest Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)
