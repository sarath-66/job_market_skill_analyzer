"""
Job Market Skill Demand Analysis
Reads cleaned_jobs.csv and produces insights:
- Top skills overall and by role
- Top locations by demand
- Skill co-occurrence (which skills appear together)
- Simple salary insights (for the subset that has salary data)

Usage:
    python analyze_jobs.py
"""

import pandas as pd
from collections import Counter
import ast

INPUT_PATH = "../data/cleaned_jobs.csv"


def load_data():
    df = pd.read_csv(INPUT_PATH)
    # skills_found was saved as a string like "['python', 'sql']" - convert back to list
    df["skills_found"] = df["skills_found_str"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) and x != "" else []
    )
    return df


def top_skills_overall(df, top_n=15):
    all_skills = [skill for skills in df["skills_found"] for skill in skills]
    counts = Counter(all_skills)
    print(f"\n=== Top {top_n} Skills Overall ===")
    for skill, count in counts.most_common(top_n):
        pct = (count / len(df)) * 100
        print(f"  {skill:25s} {count:4d} postings ({pct:.1f}%)")
    return counts


def top_skills_by_role(df, top_n=8):
    print("\n=== Top Skills By Role ===")
    for role in df["search_role"].unique():
        subset = df[df["search_role"] == role]
        all_skills = [skill for skills in subset["skills_found"] for skill in skills]
        counts = Counter(all_skills)
        print(f"\n  -- {role} ({len(subset)} postings) --")
        for skill, count in counts.most_common(top_n):
            pct = (count / len(subset)) * 100
            print(f"    {skill:25s} {count:4d} ({pct:.1f}%)")


def top_locations(df, top_n=10):
    print(f"\n=== Top {top_n} Locations ===")
    counts = df["location"].value_counts().head(top_n)
    for loc, count in counts.items():
        print(f"  {loc:30s} {count} postings")


def salary_insights(df):
    salaried = df.dropna(subset=["salary_min", "salary_max"])
    print(f"\n=== Salary Insights (based on {len(salaried)} of {len(df)} postings with disclosed salary) ===")
    if len(salaried) == 0:
        print("  No salary data available.")
        return
    for role in salaried["search_role"].unique():
        subset = salaried[salaried["search_role"] == role]
        if len(subset) == 0:
            continue
        avg_min = subset["salary_min"].mean()
        avg_max = subset["salary_max"].mean()
        print(f"  {role:25s} avg range: {avg_min:,.0f} - {avg_max:,.0f} (n={len(subset)})")
    print("\n  NOTE: this is based on a small disclosed subset - not representative of the full market.")


def skill_gap_check(df, my_skills):
    """Compare your current skills against the most in-demand skills found."""
    all_skills = [skill for skills in df["skills_found"] for skill in skills]
    counts = Counter(all_skills)
    top_skills = [s for s, _ in counts.most_common(15)]

    my_skills_lower = [s.lower() for s in my_skills]
    have = [s for s in top_skills if s in my_skills_lower]
    missing = [s for s in top_skills if s not in my_skills_lower]

    print("\n=== Skill Gap Check ===")
    print(f"  Skills you have that are in-demand: {have}")
    print(f"  Top in-demand skills you're missing: {missing}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} cleaned postings\n")

    top_skills_overall(df)
    top_skills_by_role(df)
    top_locations(df)
    salary_insights(df)

    # Edit this list to reflect your actual current skillset
    my_current_skills = [
        "python", "java", "sql", "machine learning", "deep learning",
        "pandas", "numpy", "scikit-learn", "statistics", "git",
        "tensorflow"
    ]
    skill_gap_check(df, my_current_skills)


if __name__ == "__main__":
    main()