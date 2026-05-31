import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="centered",
)

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("best_model_rf.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("selected_features__1_.pkl")
    return model, scaler, features

model, scaler, selected_features = load_artifacts()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🎗️ Breast Cancer Survival Predictor")
st.markdown(
    "Fill in the patient's clinical information below and click **Predict** "
    "to estimate survival status using a tuned Random Forest model."
)
st.divider()

# ── Input form ──────────────────────────────────────────────────────────────
st.subheader("Patient Clinical Features")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at Diagnosis", min_value=20, max_value=100, value=55)

    surgery_map = {"Mastectomy": 0, "Breast Conserving": 1}
    surgery = st.selectbox("Type of Breast Surgery", list(surgery_map.keys()))

    pam50_map = {
        "Basal": 0, "claudin-low": 1, "Her2": 2,
        "LumA": 3, "LumB": 4, "NC": 5, "Normal": 6
    }
    pam50 = st.selectbox("Pam50 + Claudin-low Subtype", list(pam50_map.keys()))

    cohort = st.number_input("Cohort", min_value=1, max_value=5, value=1)

    grade_map = {"Grade 1 (Low)": 1, "Grade 2 (Intermediate)": 2, "Grade 3 (High)": 3}
    grade = st.selectbox("Neoplasm Histologic Grade", list(grade_map.keys()))

    meno_map = {"Pre": 0, "Post": 1}
    meno = st.selectbox("Inferred Menopausal State", list(meno_map.keys()))

    integrative_map = {str(i): i for i in range(1, 11)}
    integrative = st.selectbox("Integrative Cluster", list(integrative_map.keys()))

with col2:
    laterality_map = {"Left": 0, "Right": 1}
    laterality = st.selectbox("Primary Tumor Laterality", list(laterality_map.keys()))

    lymph_nodes = st.number_input(
        "Lymph Nodes Examined Positive", min_value=0, max_value=30, value=0
    )

    mutation_count = st.number_input(
        "Mutation Count", min_value=0, max_value=500, value=50
    )

    npi = st.number_input(
        "Nottingham Prognostic Index (NPI)", min_value=0.0, max_value=10.0,
        value=4.0, step=0.1
    )

    radio = st.selectbox("Radio Therapy", ["No", "Yes"])

    relapse_map = {"Not Recurred": 0, "Recurred": 1}
    relapse = st.selectbox("Relapse Free Status", list(relapse_map.keys()))

    tumor_size = st.number_input(
        "Tumor Size (mm)", min_value=1, max_value=150, value=25
    )

st.divider()

# ── Compute engineered feature ───────────────────────────────────────────────
grade_val  = grade_map[grade]
high_grade_lymph_risk = round(grade_val * (lymph_nodes + 1), 2)

st.caption(f"📊 Computed **High Grade Lymph Risk** score: `{high_grade_lymph_risk}`")

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Survival Status", use_container_width=True, type="primary"):

    input_dict = {
        "Age at Diagnosis":           age,
        "Type of Breast Surgery":     surgery_map[surgery],
        "Pam50 + Claudin-low subtype": pam50_map[pam50],
        "Cohort":                     cohort,
        "Neoplasm Histologic Grade":  grade_val,
        "Inferred Menopausal State":  meno_map[meno],
        "Integrative Cluster":        int(integrative_map[integrative]),
        "Primary Tumor Laterality":   laterality_map[laterality],
        "Lymph nodes examined positive": lymph_nodes,
        "Mutation Count":             mutation_count,
        "Nottingham prognostic index": npi,
        "Radio Therapy":              1 if radio == "Yes" else 0,
        "Relapse Free Status":        relapse_map[relapse],
        "Tumor Size":                 tumor_size,
        "High_Grade_Lymph_Risk":      high_grade_lymph_risk,
    }

    input_df = pd.DataFrame([input_dict])[selected_features]

    input_scaled = scaler.transform(input_df)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("🟢 **Living** — The model predicts the patient is likely to survive.")
    else:
        st.error("🔴 **Deceased** — The model predicts a high mortality risk.")

    living_pct  = round(probability[0] * 100, 1)
    deceased_pct = round(probability[1] * 100, 1)

    col_a, col_b = st.columns(2)
    col_a.metric("Probability: Living",  f"{living_pct}%")
    col_b.metric("Probability: Deceased", f"{deceased_pct}%")

    st.progress(int(living_pct))

    st.info(
        "⚠️ This tool is for **educational / research purposes only** "
        "and should not replace professional medical judgment."
    )
