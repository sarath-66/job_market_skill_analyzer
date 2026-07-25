"""
Job Market Insights Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Job Market Skill Analyzer", layout="wide")

INPUT_PATH = "../data/cleaned_jobs.csv"

# ---- Edit this to match your real skillset ----
MY_SKILLS = [
    "python", "java", "sql", "machine learning", "deep learning",
    "pandas", "numpy", "scikit-learn", "statistics", "git", "tensorflow"
]


@st.cache_data
def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["skills_found"] = df["skills_found_str"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) and x != "" else []
    )
    return df


def get_skill_counts(df, role=None):
    subset = df if role is None else df[df["search_role"] == role]
    all_skills = [skill for skills in subset["skills_found"] for skill in skills]
    return Counter(all_skills), len(subset)


def main():
    st.title("📊 Job Market Skill Analyzer")
    st.caption("Self-collected data from live job postings via Adzuna API — Data Analyst, ML Engineer, Data Scientist roles (India)")

    df = load_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Postings", len(df))
    col2.metric("Roles Covered", df["search_role"].nunique())
    col3.metric("Locations", df["location"].nunique())

    st.divider()

    # --- Top skills overall ---
    st.subheader("Top Skills Overall")
    counts, total = get_skill_counts(df)
    top_skills_df = pd.DataFrame(counts.most_common(15), columns=["Skill", "Count"])
    top_skills_df["% of postings"] = (top_skills_df["Count"] / total * 100).round(1)
    st.bar_chart(top_skills_df.set_index("Skill")["Count"])
    st.dataframe(top_skills_df, width="stretch")

    st.divider()

    # --- Skills by role ---
    st.subheader("Top Skills by Role")
    role = st.selectbox("Select role", df["search_role"].unique())
    role_counts, role_total = get_skill_counts(df, role)
    role_df = pd.DataFrame(role_counts.most_common(10), columns=["Skill", "Count"])
    role_df["% of postings"] = (role_df["Count"] / role_total * 100).round(1)
    st.bar_chart(role_df.set_index("Skill")["Count"])

    st.divider()

    # --- Top locations ---
    st.subheader("Top Locations by Demand")
    loc_counts = df["location"].value_counts().head(10)
    st.bar_chart(loc_counts)

    st.divider()

    # --- Salary insights ---
    st.subheader("Salary Insights (limited disclosed data)")
    salaried = df.dropna(subset=["salary_min", "salary_max"])
    st.caption(f"Based on {len(salaried)} of {len(df)} postings that disclosed salary — not fully representative.")
    if len(salaried) > 0:
        salary_summary = salaried.groupby("search_role")[["salary_min", "salary_max"]].mean().round(0)
        st.dataframe(salary_summary, width="stretch")

    st.divider()

    # --- Skill gap ---
    st.subheader("Your Skill Gap Analysis")
    top_15 = [s for s, _ in counts.most_common(15)]
    have = [s for s in top_15 if s in MY_SKILLS]
    missing = [s for s in top_15 if s not in MY_SKILLS]

    gap_col1, gap_col2 = st.columns(2)
    with gap_col1:
        st.success("✅ Skills you have (in-demand)")
        for s in have:
            st.write(f"- {s}")
    with gap_col2:
        st.warning("⚠️ In-demand skills you're missing")
        for s in missing:
            st.write(f"- {s}")


if __name__ == "__main__":
    main()