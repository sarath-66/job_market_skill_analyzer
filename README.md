# Job Market Skill Analyzer

A self-collected data project that analyzes live job postings to find in-demand skills and predict job role categories — built to understand the real skill gaps for Data Analyst / ML Engineer / Data Scientist roles in the Indian job market.

🔗 **Live Dashboard:** [jobmarketskillanalyzer.streamlit.app](https://jobmarketskillanalyzer-fxwddykgr8wt67xf9hu3zs.streamlit.app/)
🔗 **Live Model Demo:** [Job Role Classifier](https://jobmarketskillanalyzer-jseql86aygaic4rpbkh5kk.streamlit.app/)

---

## What This Project Does

1. **Collects real, live job posting data** via the [Adzuna API](https://developer.adzuna.com/) — not a static Kaggle dataset
2. **Cleans and processes** the data, handling missing values and duplicates
3. **Extracts in-demand skills** from job descriptions using keyword matching
4. **Visualizes insights** — top skills overall, top skills by role, top hiring locations, and a personal skill-gap comparison
5. **Trains a text classifier** (SVM + TF-IDF) to predict the job role category from description text alone
6. **Analyzes model errors** to understand *why* the model gets things wrong — not just how often

---

## Dataset

- **750 job postings** collected across three roles: Data Analyst, Machine Learning Engineer, Data Scientist (India)
- After deduplication: **682 postings** used for analysis and modeling
- Source: Adzuna API (real-time job listings)

---

## Key Decision: Why Not Salary Prediction?

Salary was disclosed in only **~32% of postings** (239 / 750) — too sparse to train a reliable prediction model on. Rather than force a weak model on incomplete data, I pivoted to building a **role classifier** using the job description text, which was **fully populated** across all postings. This was a deliberate data-quality decision, not a shortcut.

Salary data is still shown in the dashboard as a side-insight, clearly labeled as based on limited disclosed data — not used for any predictive claims.

---

## Key Findings

**Top in-demand skills** (across all roles): Machine Learning, Python, SQL, Excel, Git, Power BI, Data Visualization

**Skill gap analysis** (comparing my current skillset against top 15 in-demand skills):
- ✅ Covered: Machine Learning, Python, SQL, Git, Deep Learning, Statistics
- ⚠️ Missing: Excel, Power BI, Tableau, Data Visualization, Spark, Scala, Big Data, NLP, Communication

This directly informed a learning priority: **BI/visualization tools (Power BI, Tableau)** are a bigger gap for analyst-track roles than deep ML skills.

**Role classifier performance:** ~80% accuracy (SVM with TF-IDF, unigrams + bigrams)

**Error analysis finding:** *Data Scientist* is the hardest role to classify correctly, most frequently confused with *ML Engineer*. This likely reflects a genuine pattern in the Indian job market — companies often use "Data Scientist" and "ML Engineer" titles interchangeably for very similar responsibilities — rather than indicating a weak model.

---

## Project Structure

```
job_market_project/
├── data/
│   ├── raw_jobs.csv              # raw collected data
│   ├── cleaned_jobs.csv          # cleaned + skill-tagged data
│   ├── role_classifier.pkl       # trained SVM model
│   └── tfidf_vectorizer.pkl      # fitted TF-IDF vectorizer
├── scripts/
│   ├── collect_jobs.py           # pulls data from Adzuna API
│   ├── clean_jobs.py             # cleaning + skill extraction
│   ├── analyze_jobs.py           # terminal-based analysis
│   ├── train_model.py            # trains + evaluates the classifier
│   ├── dashboard.py               # Streamlit insights dashboard
│   └── model_demo.py              # Streamlit live model demo
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Data collection:** Adzuna API, `requests`
- **Data processing:** `pandas`
- **Modeling:** `scikit-learn` (TF-IDF, SVM, Logistic Regression)
- **Visualization / deployment:** `Streamlit`, Streamlit Community Cloud

---

## Running Locally

```bash
git clone https://github.com/sarath-66/job_market_skill_analyzer.git
cd job_market_skill_analyzer
pip install -r requirements.txt

# Set your own Adzuna API credentials before collecting new data
export ADZUNA_APP_ID="your_app_id"
export ADZUNA_APP_KEY="your_app_key"

python scripts/collect_jobs.py
python scripts/clean_jobs.py
python scripts/train_model.py

# Run the dashboards
streamlit run scripts/dashboard.py
streamlit run scripts/model_demo.py
```

---

## Next Steps

- Expand the skill keyword list for more granular extraction
- Collect postings over time to track how skill demand shifts
- Close the identified gap by learning Power BI / Tableau basics

