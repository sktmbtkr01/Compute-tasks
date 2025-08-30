import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("asthma_model.pkl")
xgb_model = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]

st.title("🫁 Asthma Prediction Tool")
st.write("Fill in the patient details below to predict the likelihood of asthma.")

st.markdown("### 🧑 Patient Information")
age = st.slider("**Age**", 1, 90, 30)
gender = st.radio("**Gender**", ["Male", "Female"])


use_bmi_calc = st.checkbox("**I don’t know my BMI, calculate it**")
if use_bmi_calc:
    weight = st.number_input("**Weight (kg)**", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    height = st.number_input("**Height (cm)**", min_value=120.0, max_value=220.0, value=170.0, step=1.0)
    bmi = weight / ((height/100)**2)
    st.info(f"Your BMI is: {bmi:.2f}")
else:
    bmi = st.number_input("**BMI (Body Mass Index)**", min_value=15.0, max_value=45.0, value=25.0, step=0.1)

st.markdown("### 🚬 Lifestyle Factors")
smoking_status = st.selectbox("**Smoking Status**", ["Never", "Former", "Current"])
pollution = st.selectbox("**Air Pollution Level**", ["Low", "Moderate", "High"])
activity = st.selectbox("**Physical Activity Level**", ["Sedentary", "Moderate", "Active"])
occupation = st.selectbox("**Occupation Type**", ["Indoor", "Outdoor"])
    
st.markdown("### 🩺 Medical History")
family_history = st.radio("**Family History of Asthma?**", ["Yes", "No"])
family_history = 1 if family_history == "Yes" else 0
allergies = st.selectbox("**Allergies**", ["None", "Dust", "Multiple", "Unknown"])
comorbidities = st.selectbox("**Comorbidities**", ["None", "Diabetes", "Hypertension", "Both"])
med_adherence = st.slider("**Medication Adherence (0 = Never, 1 = Always)**", 0.0, 1.0, 0.5, step=0.01)

st.markdown("### 🫀 Clinical Measurements")
er_visits = st.number_input("**Number of ER Visits**", min_value=0, max_value=6, value=1, step=1)
pef = st.number_input("**Peak Expiratory Flow (L/min)**", min_value=150.0, max_value=600.0, value=400.0, step=1.0)
feno = st.number_input("**FeNO Level (ppb)**", min_value=5.0, max_value=65.0, value=25.0, step=0.1)

input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Smoking_Status": smoking_status,
    "Family_History": family_history,
    "Allergies": allergies,
    "Air_Pollution_Level": pollution,
    "Physical_Activity_Level": activity,
    "Occupation_Type": occupation,
    "Comorbidities": comorbidities,
    "Medication_Adherence": med_adherence,
    "Number_of_ER_Visits": er_visits,
    "Peak_Expiratory_Flow": pef,
    "FeNO_Level": feno
}])

if st.button("🔮 Predict"):
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ Asthma Detected (Probability {probs[1]*100:.2f}%)")
    else:
        st.success(f"✅ No Asthma detected (Probability {probs[0]*100:.2f}%)")

    st.write("### Prediction Confidence")
    st.progress(int(probs[1]*100))

    st.write("### 🔍 Top Factors Influencing Prediction (Global Importance)")
    importances = xgb_model.feature_importances_
    features = preprocessor.get_feature_names_out()
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax, palette="viridis")
    st.pyplot(fig)
