"""
Job Role Classifier - Live Demo
Paste a job description, get a predicted role category.

Usage:
    streamlit run model_demo.py
"""

import streamlit as st
import joblib
import re

st.set_page_config(page_title="Job Role Classifier Demo", layout="centered")

MODEL_PATH = "../data/role_classifier.pkl"
VECTORIZER_PATH = "../data/tfidf_vectorizer.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def main():
    st.title("🎯 Job Role Classifier")
    st.caption(
        "Trained on 682 self-collected job postings via the Adzuna API. "
        "Paste any job description below and the model predicts whether it's a "
        "Data Analyst, Machine Learning Engineer, or Data Scientist role — based on text alone."
    )

    model, vectorizer = load_model()

    example = st.selectbox(
        "Try an example, or paste your own below:",
        [
            "-- write your own --",
            "Looking for a candidate skilled in SQL, Excel, and Power BI to build dashboards and generate business insights for stakeholders.",
            "We need someone experienced in deep learning, PyTorch, and model deployment to build and scale production ML pipelines.",
            "Seeking a candidate with strong statistics background to design experiments, build predictive models, and communicate findings to leadership.",
        ],
    )

    default_text = "" if example == "-- write your own --" else example
    description = st.text_area("Job description text:", value=default_text, height=180)

    if st.button("Predict Role", type="primary"):
        if not description.strip():
            st.warning("Please paste a job description first.")
            return

        cleaned = clean_text(description)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        st.success(f"**Predicted Role:** {prediction.title()}")

        # Show decision confidence if the model supports probability/decision scores
        if hasattr(model, "decision_function"):
            scores = model.decision_function(vec)[0]
            classes = model.classes_
            score_dict = dict(zip(classes, scores))
            st.subheader("Confidence Scores (raw decision values)")
            for cls, score in sorted(score_dict.items(), key=lambda x: -x[1]):
                st.write(f"- {cls.title()}: {score:.3f}")

    st.divider()
    st.caption(
        "Model: SVM with TF-IDF features (unigrams + bigrams), ~80% test accuracy. "
        "Data Scientist is the hardest class to predict correctly - likely reflects real overlap "
        "in how these job titles are used in the Indian job market."
    )


if __name__ == "__main__":
    main()