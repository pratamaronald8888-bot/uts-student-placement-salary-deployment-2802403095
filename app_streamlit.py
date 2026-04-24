from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

CLASSIFICATION_MODEL_CANDIDATES = [
    Path("artifacts/student_placement_classifier.pkl"),
    Path("student_placement_classifier.pkl"),
]
REGRESSION_MODEL_CANDIDATES = [
    Path("artifacts/student_salary_regressor.pkl"),
    Path("student_salary_regressor.pkl"),
]


def resolve_model_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def predict_locally(data):
    classification_model_path = resolve_model_path(CLASSIFICATION_MODEL_CANDIDATES)
    regression_model_path = resolve_model_path(REGRESSION_MODEL_CANDIDATES)

    if classification_model_path is None or regression_model_path is None:
        st.error("Model belum ada. Jalankan `python pipeline_student.py` dulu.")
        return None

    classification_model = joblib.load(classification_model_path)
    regression_model = joblib.load(regression_model_path)
    input_data = pd.DataFrame([data])

    return {
        "placement_status": str(classification_model.predict(input_data)[0]),
        "salary_lpa": round(float(regression_model.predict(input_data)[0]), 2),
    }


def main():
    st.set_page_config(page_title="Student Model Deployment", layout="centered")

    st.title("Student Placement and Salary Prediction")
    st.subheader("Model Deployment")

    gender = st.selectbox("gender", ["Male", "Female"])
    branch = st.selectbox("branch", ["CSE", "IT", "ECE", "ME", "Civil"])
    cgpa = st.number_input("cgpa", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    tenth_percentage = st.number_input("tenth_percentage", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    twelfth_percentage = st.number_input("twelfth_percentage", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    backlogs = st.number_input("backlogs", min_value=0, max_value=20, value=0, step=1)
    study_hours_per_day = st.number_input("study_hours_per_day", min_value=0.0, max_value=24.0, value=4.0, step=0.1)
    attendance_percentage = st.number_input("attendance_percentage", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    projects_completed = st.number_input("projects_completed", min_value=0, max_value=30, value=3, step=1)
    internships_completed = st.number_input("internships_completed", min_value=0, max_value=10, value=1, step=1)
    coding_skill_rating = st.slider("coding_skill_rating", 1, 10, 6)
    communication_skill_rating = st.slider("communication_skill_rating", 1, 10, 6)
    aptitude_skill_rating = st.slider("aptitude_skill_rating", 1, 10, 6)
    hackathons_participated = st.number_input("hackathons_participated", min_value=0, max_value=30, value=1, step=1)
    certifications_count = st.number_input("certifications_count", min_value=0, max_value=30, value=2, step=1)
    sleep_hours = st.number_input("sleep_hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    stress_level = st.slider("stress_level", 1, 10, 5)
    part_time_job = st.selectbox("part_time_job", ["No", "Yes"])
    family_income_level = st.selectbox("family_income_level", ["Low", "Medium", "High"])
    city_tier = st.selectbox("city_tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet_access = st.selectbox("internet_access", ["Yes", "No"])
    extracurricular_involvement = st.selectbox("extracurricular_involvement", ["Low", "Medium", "High"])

    data = {
        "gender": gender,
        "branch": branch,
        "cgpa": float(cgpa),
        "tenth_percentage": float(tenth_percentage),
        "twelfth_percentage": float(twelfth_percentage),
        "backlogs": int(backlogs),
        "study_hours_per_day": float(study_hours_per_day),
        "attendance_percentage": float(attendance_percentage),
        "projects_completed": int(projects_completed),
        "internships_completed": int(internships_completed),
        "coding_skill_rating": int(coding_skill_rating),
        "communication_skill_rating": int(communication_skill_rating),
        "aptitude_skill_rating": int(aptitude_skill_rating),
        "hackathons_participated": int(hackathons_participated),
        "certifications_count": int(certifications_count),
        "sleep_hours": float(sleep_hours),
        "stress_level": int(stress_level),
        "part_time_job": part_time_job,
        "family_income_level": family_income_level,
        "city_tier": city_tier,
        "internet_access": internet_access,
        "extracurricular_involvement": extracurricular_involvement,
    }

    if st.button("Make Prediction"):
        result = predict_locally(data)
        if result:
            st.success("Prediction completed")
            st.write(f"Placement Prediction: **{result['placement_status']}**")
            st.write(f"Predicted Salary: **{result['salary_lpa']:.2f} LPA**")


if __name__ == "__main__":
    main()
