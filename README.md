# hnn-textanalysis
This repo is dedicated to a series of scripts meant to enrich a CSV of analytics on Sneakers articles written on Hot New Hip Hop. At a high level, the program scrapes content from HNNH, analyzes the text with GPT, creates embeddings and builds a neural network (intended) to predict the # of views an article might get.

The pipeline:

1. Crawls and extracts only my roommate's articles from the company's webpage using the **Firecrawl API** and URLs in the provided CSV
2. The content is then processed with GPT to generate some basic features like the article type and shoe brand mentioned. Furthermore some text attributes are calculated (number of paragraphs, article length, etc.). Embeddings are also created.
3. Using PyTorch a Neural Network is then trained, cross-validated and tested.
4. This model is then exposed through a simple Streamlit app allowing a user to input an article and receive a prediction of the article's page views.

---

## Initial Goals

- **Web Scraping & Ingestion**
  - Use Firecrawl API to crawl and extract article content from the web.
  - Clean the scraped data and format the text for downstream processing.
- **Data Engineering**
  - Combine scraped data to the provided CSV of analytics
  - Add additioanl features to the data set using OpenAI embeddings, GPT and basic Python computations.
  - Produce a structured dataset suitable for ML and exploratory analysis.
- **Machine Learning**
  - Apply ML techniques (informed by QTM-3635) to:
    - Predict artiicle performance/page views.
    - Identify which features matter most (e.g., text features, metadata, publish time).
- **Visualization / WebApp**
  - Build a simple dashboard/web app to make the model usable to a user seeking to 'optimize' their text.

---

## Data

Currently available metrics provided by an external analytics source include (per article):

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

## Tech Stack

- **Language:** Python
- **Scraping:** Firecrawl API
- **Data / Analysis:** pandas, NumPy, scikit-learn (and/or similar)
- **Modeling:** Regression / classification models; feature importance analysis
- **Dashboard / WebApp:** (e.g., Streamlit / FastAPI + frontend – TBD)

### Required APIs:
Users will need an [OpenAI](https://openai.com/api/pricing/) & [Firecrawl API](https://www.firecrawl.dev/pricing) key. The Firecrawl API can be used for free (with 500 limited credits). 

---

## Project Structure (Proposed)

```
.
├── data/
│   └── hnnh_base.csv             # Base CSV File to be enriched and GPT Processed
│   
├── src/
│   ├── models/
│   │   │── feature_engineering.py       # Creating new features using OpenAI API and Python computations
│   │   │── prompts/
│   │   │   └── prompt1.py               # Prompt used to extract sneaker brand, article type and sneaker price from article
│   │   │── model1.py                    # Python program using PyTorch to create and run a Neural Network using the engineered features
│   │   └── EDA.ipynb                    # Jupyter notebook dedicated to Exploratory Data Analysis
│   ├── scraping/
│   │   └── firecrawl_client.py   # Wrapper around Firecrawl API
│
├── app.py                        # Streamlit App exposing underlying Neural Network
├── data_dictionary.md            # Dictionary explaining the column headers of the base CSV
├── README.md
└── requirements.txt
```
## The Data:

## How To Use These Scripts:
If you want to test the full data processing pipeline I reccomend you delete all CSVs other than: [this CSV](data/hnnh_base.csv).

## Code Explanation:
**Note: The EDA notebook is 'read only', none of the changes are saved to the HNHH_Processed.CSV. Instead I decided to make these changes in the model scripts instead. This allows the Model scripts to be run without having to use the Jupyter Notebook.** 

## Considering why this might not have worked?
Due to the presence of outliers with a large # of pageviews I decided to use a log scale in the response.