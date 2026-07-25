"""
Job Market Data Collector
Pulls job postings from Adzuna API for a given role + country, saves to CSV.

Usage:
    python collect_jobs.py

Before running: set your APP_ID and APP_KEY below (from https://developer.adzuna.com/)
"""

import requests
import pandas as pd
import time
import os

# ---- CONFIG: fill these in after you register ----
APP_ID = "3a1fdfd4"
APP_KEY = "cbd63746603f4f534e69768c43bf07af"
COUNTRY = "in"  # India
ROLES = ["data analyst", "machine learning engineer", "data scientist"]
PAGES_PER_ROLE = 5  # ~50 results per page, so 5 pages ≈ 250 postings per role
OUTPUT_PATH = "data/raw_jobs.csv"

BASE_URL = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search"


def fetch_jobs_for_role(role, pages=PAGES_PER_ROLE):
    """Fetch job postings for a single role across multiple pages."""
    all_jobs = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/{page}"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "what": role,
            "results_per_page": 50,
            "content-type": "application/json",
        }

        # Retry up to 3 times on network errors
        resp = None
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                break  # success, exit retry loop
            except requests.exceptions.RequestException as e:
                print(f"  [!] Attempt {attempt}/3 failed for '{role}' page {page}: {e}")
                if attempt < 3:
                    wait = 2 ** attempt  # exponential backoff: 2s, 4s
                    print(f"      Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [!] Giving up on '{role}' page {page} after 3 attempts.")

        if resp is None:
            continue  # skip this page if all retries failed

        if resp.status_code != 200:
            print(f"  [!] Error fetching page {page} for '{role}': {resp.status_code} - {resp.text[:200]}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            print(f"  [!] No more results for '{role}' at page {page}, stopping early.")
            break

        for job in results:
            all_jobs.append({
                "search_role": role,
                "title": job.get("title"),
                "company": job.get("company", {}).get("display_name"),
                "location": job.get("location", {}).get("display_name"),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
                "description": job.get("description"),
                "category": job.get("category", {}).get("label"),
                "contract_type": job.get("contract_type"),
                "posted_date": job.get("created"),
                "url": job.get("redirect_url"),
            })

        print(f"  [{role}] page {page}: {len(results)} jobs collected")
        time.sleep(1)  # be polite to the API

    return all_jobs


def main():
    print("Starting job collection...\n")
    all_jobs = []

    for role in ROLES:
        print(f"Fetching role: {role}")
        jobs = fetch_jobs_for_role(role)
        all_jobs.extend(jobs)
        print(f"  -> Total so far: {len(all_jobs)}\n")

    df = pd.DataFrame(all_jobs)
    print(f"Total jobs collected: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()