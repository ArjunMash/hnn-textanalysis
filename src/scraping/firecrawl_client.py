import os
from dotenv import load_dotenv
from firecrawl import Firecrawl
from pydantic import BaseModel
from bs4 import BeautifulSoup
from pprint import pprint

import requests
import json

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# Define clients:
#firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)

def get_html(url:str)->dict: 
    """
    Extract the HTML for the main text for each page at the given url and return it 
    """
    fc_api_url = "https://api.firecrawl.dev/v2/scrape"

    payload = {
    "url": url,
    "onlyMainContent": True, # eliminates elements like navbar
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
    response = response.json()
    html = response["data"]["html"]

    return(html)

def enrich_csv():
    """
    Iterate across csv enriching with text For each csv first check if it's already been enriched. Add colun with request key statuus. So if != 200 then enrich. And print list when error occurs

    Write to *enriched.csv
    """


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Dek / intro sentence (the bold line atop each article)
    dek_span = soup.select_one(
        "span.font-lora.text-dark-grey.font-bold, "
        "span.font-lora.text-dark-grey.font-extrabold"
    )
    parts = []
    if dek_span:
        dek_text = dek_span.get_text(strip=True)
        if dek_text:
            parts.append(dek_text)

    # 2. Main article body
    article = soup.select_one("article")
    if not article:
        # fallback: this site is pretty consistent, so if no article is found, just bail early
        return "\n\n".join(parts)

    # gather all paragraph texts inside the article
    for p in article.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:
            continue

        # skip obvious junk blocks you don't want as "article text"
        if text.startswith("Read More:"):
            continue
        if "VIDEO" in text:
            continue

        parts.append(text)

    return "\n\n".join(parts)


def append_text_article

def main():
    # Start off with one URL and get extraction workflow down
    url =  "https://www.hotnewhiphop.com/952232-air-jordan-11-285-sneaker-news-2"
    article = get_html(url)
    # article_text = article
    # print(article)
    article_text, article_meta = extract_article_text(article)
    print(article_text)
    
    


if __name__ == "__main__":
    main()





