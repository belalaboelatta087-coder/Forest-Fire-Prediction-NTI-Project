import streamlit as st

# ============================================================
# PAGE CONFIGURATION (Must be the very first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Forest Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# FOREST FIRE RISK PREDICTION
# Streamlit Application
# ============================================================

import joblib
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.title("🔥 تطبيق التنبؤ بحدوث الحرائق")

# ============================================================
# LOAD FINAL MODELS
# ============================================================

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Path to the model file
MODEL_PATH = BASE_DIR / "forest_fire_final_model.pkl"

# Check if the model file exists
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

# Load the model configuration
try:
    model_config = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================
# EXTRACT MODELS AND SETTINGS
# ============================================================

model = model_config["model"]
dt_model = model_config["dt_model"]
svm_model = model_config["svm_model"]
features = model_config["features"]
threshold = model_config["threshold"]

st.success("تم تحميل النموذج بنجاح!")
