"""
Job Role Classifier
Predicts whether a job posting is for: data analyst, machine learning engineer,
or data scientist — based on the job description text.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

INPUT_PATH = "../data/cleaned_jobs.csv"
MODEL_OUTPUT = "../data/role_classifier.pkl"
VECTORIZER_OUTPUT = "../data/tfidf_vectorizer.pkl"


def load_data():
    df = pd.read_csv(INPUT_PATH)
    # description_clean already has lowercased, tag-stripped text from clean_jobs.py
    df = df.dropna(subset=["description_clean", "search_role"])
    return df


def train_and_evaluate(df):
    X = df["description_clean"]
    y = df["search_role"]

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{y.value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorization - converts text into numeric features
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2)  # unigrams + bigrams (e.g. "machine learning" as one feature)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Try two models, compare - matches your usual approach (LR/SVM comparison)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="linear", probability=True),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, preds, acc)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds))

    # Pick best model
    best_name = max(results, key=lambda k: results[k][2])
    best_model, best_preds, best_acc = results[best_name]
    print(f"\nBest model: {best_name} (accuracy: {best_acc:.3f})")

    # --- Error analysis: where does it go wrong? ---
    print("\n=== Confusion Matrix (best model) ===")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    print("\n=== Sample Misclassifications ===")
    test_df = pd.DataFrame({"actual": y_test.values, "predicted": best_preds})
    mistakes = test_df[test_df["actual"] != test_df["predicted"]]
    print(f"Total mistakes: {len(mistakes)} out of {len(test_df)}")
    if len(mistakes) > 0:
        print(mistakes.head(10))

    # Save model + vectorizer for later use (e.g. in a demo)
    joblib.dump(best_model, MODEL_OUTPUT)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT)
    print(f"\nSaved model to {MODEL_OUTPUT}")
    print(f"Saved vectorizer to {VECTORIZER_OUTPUT}")

    return best_model, vectorizer


def main():
    df = load_data()
    train_and_evaluate(df)


if __name__ == "__main__":
    main()