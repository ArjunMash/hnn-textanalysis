# hnn-textanalysis
Scraping content from HNN and conducting ML modelling on each article

# Sneaker Article Performance Analytics

This project analyzes the performance of sneaker articles written by a single author (~1,543 articles over the past year) and explores which content and metadata features are associated with higher engagement and revenue.

The pipeline:

1. Crawls and extracts only my roommate's articles from the company's webpage using the **Firecrawl API**.
2. Structures the content and engagement metrics into a clean dataset for analysis.
3. Trains and evaluates ML models to understand what drives article performance.
4. Exposes results through a simple dashboard / web app.

---

## Goals

- **Web Scraping & Ingestion**
  - Use Firecrawl API to crawl and extract article content and metadata for a single author.
  - Deduplicate, clean, and normalize article data.
- **Data Engineering**
  - Combine scraped data with external analytics metrics.
  - Produce a structured dataset suitable for ML and exploratory analysis.
- **Machine Learning**
  - Apply ML techniques (informed by QTM-3635) to:
    - Predict engagement / performance.
    - Identify which features matter most (e.g., text features, metadata, publish time).
- **Visualization / WebApp**
  - Build a simple dashboard/web app to explore:
    - Article-level metrics.
    - Feature importance.
    - Distributions and trends over time.

---

## Data

Currently available metrics from an external analytics source include (per article):

- `url`
- `uniqueUsers`
- `pageViewsTotal`
- `totalEngagementTime`
- `averageScroll`
- `averageDurationPerUser`
- `AverageRFV` (Recency, Frequency, Value)
- `cumulativeRFV`
- `discoverImpressions`
- `googleNewsImpressions`
- `searchImpressions`

Plus whatever is scraped from the article pages:

- Title, subtitle
- Publish date
- Author
- Article body text
- Tags / categories (if available)
- URL, slug, etc.

---

## Tech Stack (Planned)

- **Language:** Python
- **Scraping:** Firecrawl API
- **Data / Analysis:** pandas, NumPy, scikit-learn (and/or similar)
- **Modeling:** Regression / classification models; feature importance analysis
- **Dashboard / WebApp:** (e.g., Streamlit / FastAPI + frontend – TBD)

---

## Project Structure (Proposed)

```
.
├── data/
│   ├── raw/          # Raw scraped data & analytics exports
│   └── processed/    # Cleaned, merged datasets ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── scraping/
│   │   └── firecrawl_client.py   # Wrapper around Firecrawl API
│   ├── data/
│   │   └── merge_analytics.py    # Join scraped data with metrics
│   ├── models/
│   │   └── train_models.py       # Training & evaluation scripts
│   └── app/
│       └── dashboard.py          # Web dashboard / web app entrypoint
├── .env.example
├── requirements.txt
└── README.md
```
