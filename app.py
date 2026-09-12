import streamlit as st

# ============================================================
# PAGE CONFIGURATION (Must be the very first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Forest Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FOREST FIRE RISK PREDICTION
# Streamlit Application
# ============================================================

import joblib
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.title("🔥 تطبيق التنبؤ بحدوث حرائق الغابات")
st.write("أدخل قيم الخصائص في القائمة الجانبية للتنبؤ بنسبة احتمال حدوث حريق.")

# ============================================================
# LOAD FINAL MODELS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "forest_fire_final_model.pkl"

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model_config = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================
# EXTRACT MODELS AND SETTINGS
# ============================================================

model = model_config["model"]
dt_model = model_config.get("dt_model", None)
svm_model = model_config.get("svm_model", None)
features = model_config.get("features", [])
threshold = model_config.get("threshold", 0.5)

# ============================================================
# SIDEBAR - FEATURE INPUTS
# ============================================================

st.sidebar.header("🎛️ أدخل قيم الخصائص")

input_data = {}
for feat in features:
    # إنشاء خانة إدخال رقمية لكل ميزة في النموذج
    input_data[feat] = st.sidebar.number_input(f"الميزة: {feat}", value=0.0)

input_df = pd.DataFrame([input_data])

# ============================================================
# PREDICTION & RESULTS
# ============================================================

st.subheader("📊 النتيجة والتنبؤ")

if st.button("🚀 إجراء التنبؤ الآن"):
    try:
        # حساب الاحتمالية باستخدام النموذج الأساسي (XGBoost)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        else:
            prob = float(model.predict(input_df)[0])

        is_fire = prob >= threshold

        col1, col2 = st.columns(2)

        with col1:
            if is_fire:
                st.error(f"⚠️ **خطر حريق!** (احتمالية الحريق: {prob * 100:.2f}%)")
            else:
                st.success(f"✅ **آمن / لا يوجد حريق** (احتمالية الحريق: {prob * 100:.2f}%)")

        with col2:
            # رسم مؤشر النسبة المئوية
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "نسبة الخطر (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if is_fire else "green"},
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "lightgreen"},
                        {'range': [threshold * 100, 100], 'color': "pink"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as err:
        st.error(f"حدث خطأ أثناء الحساب: {err}")
else:
    st.info("قم بضبط الخصائص من القائمة الجانبية (Sidebar) ثم اضغط على زر التنبؤ أعلاه.")
