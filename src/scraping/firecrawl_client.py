import os
from dotenv import load_dotenv
from firecrawl import Firecrawl
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
import time


# Load environment variables from .env file
load_dotenv()

# Get API key from environment
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API")

def get_html(url:str)->str:
    """
    Extract the HTML for the main text for each page at the given url and return it
    """
    fc_api_url = "https://api.firecrawl.dev/v2/scrape"

    payload = {
    "url": url,
    "onlyMainContent": True, # Firecrawl handles some of the HTML trimming. Eliminates elements like navbar
    "maxAge": 172800000,
    "parsers": [
        "pdf"
    ],
    "formats": [
        "html"
    ]
    }
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(fc_api_url, json=payload, headers=headers)

    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"API returned status {response.status_code}: {response.text}")

    response_json = response.json()

    # Check if response has expected structure
    if "data" not in response_json:
        raise Exception(f"No 'data' in response: {response_json}")

    if "html" not in response_json["data"]:
        raise Exception(f"No 'html' in data: {response_json}")

    html = response_json["data"]["html"]

    return html

def retry_failed_urls(csv_file:str):
    """
    Retry scraping for URLs that failed (have None/empty text column).
    Reads and writes to the same file.

    I built this function to be optionally added into the script in case of a high amount of failed URLs being scraped. 
    """
    df = pd.read_csv(csv_file)

    # Check if text column exists
    if 'text' not in df.columns:
        print("No 'text' column found. Run enrich_csv first.")
        return

    # Rate limiting:
    # RATE_LIMIT_DELAY = 6  # Free tier: 10 requests per minute = 6 seconds between requests
    RATE_LIMIT_DELAY = 0.6  # Paid tier: 100 requests per minute = 0.6 seconds between requests

    # Find rows with missing text
    failed_mask = df['text'].isna() | (df['text'] == '') | (df['text'] == 'None')
    failed_indices = df[failed_mask].index.tolist()

    if len(failed_indices) == 0:
        print("No failed URLs to retry!")
        return

    print(f"Found {len(failed_indices)} failed URLs to retry")

    for i, index in enumerate(failed_indices):
        url = df.at[index, 'url']

        try:
            html = get_html(url)
            article_text, article_meta = extract_article_text(html) # type: ignore

            df.at[index, 'text'] = article_text # type: ignore
            df.at[index, 'meta_description'] = article_meta # type: ignore

            print(f"Retried {i + 1}/{len(failed_indices)}: {url}")

            # Rate limiting
            if i < len(failed_indices) - 1:
                time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"Error retrying {url}: {e}")
            # Keep as None
            if i < len(failed_indices) - 1:
                time.sleep(RATE_LIMIT_DELAY)

    # Save back to same file
    df.to_csv(csv_file, index=False)
    print(f"Saved updated data to {csv_file}")

def enrich_csv(read_file:str, write_file:str):
    """
    Iterate across csv enriching with text For each csv first check if it's already been enriched. Add column with request key status. So if != 200 then enrich. And print list when error occurs

    Write to given write file
    """
    df = pd.read_csv(read_file) # start limiting it by 10 before letting it rip

    # Add columns if they don't exist
    if 'text' not in df.columns:
        df['text'] = None
    if 'meta_description' not in df.columns:
        df['meta_description'] = None

    # Rate limiting:
    # RATE_LIMIT_DELAY = 6  # Free tier: 10 requests per minute = 6 seconds between requests
    RATE_LIMIT_DELAY = 0.6  # Paid tier: 100 requests per minute = 0.6 seconds between requests

    # Checkpointing: save progress every N rows
    CHECKPOINT_INTERVAL = 50  # Save every 50 rows

    processed_count = 0

    for index, row in df.iterrows():
        url = row['url'] # type: ignore

        try:
            # Call Firecrawl
            html = get_html(url)
            article_text, article_meta = extract_article_text(html) # type: ignore

            # Store in DataFrame
            df.at[index, 'text'] = article_text # type: ignore
            df.at[index, 'meta_description'] = article_meta # type: ignore

            processed_count += 1
            print(f"Processed {processed_count}/{len(df)}: {url}") # type: ignore

            # Checkpoint: save progress every N rows
            if processed_count % CHECKPOINT_INTERVAL == 0:
                df.to_csv(write_file, index=False)
                print(f"Checkpoint: Saved progress at {processed_count}/{len(df)} rows")

            # Rate limiting: wait before next request
            if processed_count < len(df):  # Don't wait after last request
                time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"Error processing {url}: {e}")
            df.at[index, 'text'] = None # type: ignore
            df.at[index, 'meta_description'] = None # type: ignore

            processed_count += 1

            # Still wait even on error to avoid overloading API
            if processed_count < len(df):
                time.sleep(RATE_LIMIT_DELAY)

    # Final save
    df.to_csv(write_file, index=False)
    print(f"Final save: Completed enrichment of {len(df)} rows")


def extract_article_text(html: str) -> tuple:
    soup = BeautifulSoup(html, "html.parser")

    # Dek / meta description (the bold line atop each article and SEO preview)
    meta_desc = None
    dek_span = soup.select_one(
        "span.font-lora.text-dark-grey.font-bold, "
        "span.font-lora.text-dark-grey.font-extrabold"
    )
    
    if dek_span:
        dek_text = dek_span.get_text(strip=True)
        if dek_text:
            meta_desc = (dek_text)

    parts = []
    # Main article body
    article = soup.select_one("article")
    if not article:
        # fallback: this site is pretty consistent, if no article is found, return early
        return "\n\n".join(parts), meta_desc

    # gather all paragraph texts inside the article
    for p in article.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:
            continue

        # skip junk blocks not actually "article text"
        if text.startswith("Read More:"):
            continue
        if "VIDEO" in text:
            continue

        parts.append(text)

    return "\n\n".join(parts), meta_desc

def main():
    # Start off with one URL and get extraction workflow down
    #url =  "https://www.hotnewhiphop.com/952232-air-jordan-11-285-sneaker-news-2"
    # article = get_html(url)
    # article_text = article
    # print(article)
    
    # Get article body text and meta description
    #article_text, article_meta = extract_article_text(article)
    # print(article_text)
    # print(article_meta) 
    
    read_file = "/Users/arjunmasciarelli/CodingProjects/hnn-textanalysis/data/hnnh_base.csv"
    write_file = "/Users/arjunmasciarelli/CodingProjects/hnn-textanalysis/data/hnhh_enriched.csv"
    
    enrich_csv(read_file, write_file)
    

if __name__ == "__main__":
    main()
