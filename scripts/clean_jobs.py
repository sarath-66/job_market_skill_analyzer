"""
Job Data Cleaner + Skill Extractor
Reads raw_jobs.csv, cleans it, extracts skills mentioned in job descriptions,
and saves a cleaned CSV ready for analysis.

Usage:
    python clean_jobs.py
"""

import pandas as pd
import re
import os

INPUT_PATH = "../data/raw_jobs.csv"
OUTPUT_PATH = "../data/cleaned_jobs.csv"

# Common skills to search for in job descriptions (expand this list as needed)
SKILLS_LIST = [
    "python", "sql", "excel", "power bi", "tableau", "r programming",
    "machine learning", "deep learning", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "aws", "azure", "gcp", "spark", "hadoop",
    "statistics", "a/b testing", "nlp", "communication", "data visualization",
    "etl", "airflow", "docker", "git", "java", "scala", "big data",
]


def clean_data(df):
    """Basic cleaning: drop sparse columns, handle missing values, dedupe."""
    # Drop contract_type - too sparse to be useful (was ~93% missing)
    df = df.drop(columns=["contract_type"], errors="ignore")

    # Fill missing company names
    df["company"] = df["company"].fillna("Not specified")

    # Drop exact duplicate postings (same title + company + location)
    before = len(df)
    df = df.drop_duplicates(subset=["title", "company", "location"])
    after = len(df)
    print(f"Removed {before - after} duplicate postings")

    # Clean description text: remove HTML tags/extra whitespace
    df["description_clean"] = df["description"].apply(
        lambda x: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(x))).strip().lower()
    )

    return df


def extract_skills(description):
    """Return list of skills found in a description string."""
    found = []
    for skill in SKILLS_LIST:
        if skill in description:
            found.append(skill)
    return found


def main():
    print("Loading raw data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    df = clean_data(df)

    print("Extracting skills from descriptions...")
    df["skills_found"] = df["description_clean"].apply(extract_skills)
    df["skill_count"] = df["skills_found"].apply(len)

    # Convert skills list to comma-separated string for CSV storage
    df["skills_found_str"] = df["skills_found"].apply(lambda x: ", ".join(x))

    print(f"\nFinal cleaned dataset: {len(df)} rows")
    print(f"Rows with at least 1 skill detected: {(df['skill_count'] > 0).sum()}")
    print(f"Average skills detected per posting: {df['skill_count'].mean():.2f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()