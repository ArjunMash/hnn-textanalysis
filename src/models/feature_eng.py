import os
from dotenv import load_dotenv
from openai import OpenAI
import re
from typing import List, Literal, Optional
from pydantic import BaseModel
import pandas as pd

from prompts.prompt1 import SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()

text_sample = """"The Nike Air Foamposite One “Cough Drop” is making a comeback this fall. Known for its glossy black shell and bold red outsole, the iconic colorway remains one of the most recognizable in the Foamposite lineage.

Originally released over a decade ago, the sneaker's blend of innovation and aggression helped define basketball sneakers in the late 2000s and continues to have cultural relevance today. Nike introduced the Air Foamposite One in 1997, pushing the boundaries of performance and design.

Penny Hardaway’s signature shoe arrived at a time when futuristic silhouettes were taking over the court. With its one-piece molded upper and signature Zoom cushioning, the Foamposite lineup was ahead of its time.

Today, it holds cult status among collectors and hoopers alike. In-hand images reveal the same fierce energy. The all-black Foam shell is matched with suede overlays and red accents throughout.

The Nike Air Foamposite One “Cough Drop” features a black Foamposite shell with black suede overlays and mesh tongue. Also, red accents hit the outsole, heel, pull tabs, and branding.

Zoom Air cushioning and a carbon fiber plate provide support and bounce. This colorway blends aggression with legacy. A translucent red sole adds a fiery finish, while the black mesh tongue and carbon fiber shank round things out. Overall, for Foamposite fans, this Fall 2025 re-release is already building buzz.

Sneaker Bar Detroit reports that the Nike Air Foamposite One “Cough Drop” will be released at some point in the fall of 2025. Also, the retail price of the sneakers will be $240 when they are released.

The “Cough Drop” colorway originally dropped in 2010 and returned again in 2017, making this its third release. Its bold black and red combination has become one of the most beloved Foamposite looks. With nostalgia riding high, this upcoming drop is already stirring excitement among longtime fans."
"""

# Define GPT Structure:
class SneakerArticleInfo(BaseModel):
    """
    Parsed features from a sneaker-related article.

    - sneaker_price: numeric retail/sale price of the primary sneaker, or None if not present.
    - article_type: one of "sneaker_releases", "news", "features".
    - sneaker_brand: primary sneaker brand as a string, or "" if not identifiable.
    """
    sneaker_price: Optional[float]
    article_type: Literal["sneaker_releases", "news", "features"]
    sneaker_brand: str

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(body):
    """Return embeddings for provided text"""


def get_body_struct(body)-> tuple:
    """Compute avg sentence length, number of sentences and paragraphs"""
    # Count paragraphs (split by single single newlines)
    paragraphs = []
    for p in body.split('\n'):
        stripped = p.strip()
        if stripped:
            paragraphs.append(stripped)
    num_paragraphs = len(paragraphs)

    # Split by common sentence delimiters
    raw_sentences = re.split(r'[.!?]+', body)
    
    # Filter out empty strings and whitespace-only strings
    sentences = []
    for s in raw_sentences:
        stripped = s.strip()
        if stripped:
            sentences.append(stripped)
    num_sentences = len(sentences)

    # Calculate average sentence length (in words)
    if num_sentences > 0:
        total_words = 0 
        for sentence in sentences:
            total_words += len(sentence.split())
        avg_sentence_length = total_words / num_sentences
    else:
        avg_sentence_length = 0.0

    return avg_sentence_length, num_sentences, num_paragraphs, total_words

def gpt_process(body)-> tuple:
    """Uses OpenAI API and structured outputs to return the article type, primary sneaker brand and sneaker price"""
    response = client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": body,
            },
        ],
        text_format=SneakerArticleInfo,
    )
    
    info = response.output_parsed
    sneaker_price = info.sneaker_price
    article_type = info.article_type
    sneaker_brand = info.sneaker_brand
    
    return sneaker_price, article_type, sneaker_brand

    
def main():
    # # Test body structure
    # sen_len, num_s, num_p, num_w = get_body_struct(text_sample)
    # print("Avg. Sentence length is:", sen_len)
    # print("Number of sentences is:", num_s)
    # print("Number of paragraphs is:", num_p)
    # print("Number of words is:", num_w)
    
    # sneaker_price, article_type, sneaker_brand = gpt_process(text_sample)
    # print("Sneaker price is:", sneaker_price)
    # print("Article type is:", article_type)
    # print("Sneaker brand is:", sneaker_brand)
    # Load only the first 10 rows of the enriched CSV
    df = pd.read_csv('data/hnhh_enriched.csv', nrows=10)

    # Process each row
    for i in range(len(df)):
        article_text = df.loc[i, 'text']

        print(f"Processing article {i+1}/10...")

        # Get body structure features
        sen_len, num_s, num_p, num_w = get_body_struct(article_text)
        df.loc[i, 'avg_sentence_length'] = sen_len
        df.loc[i, 'num_sentences'] = num_s
        df.loc[i, 'num_paragraphs'] = num_p
        df.loc[i, 'num_words'] = num_w

        # Get GPT features
        sneaker_price, article_type, sneaker_brand = gpt_process(article_text)
        df.loc[i, 'sneaker_price'] = sneaker_price
        df.loc[i, 'article_type'] = article_type
        df.loc[i, 'sneaker_brand'] = sneaker_brand

    # Display results
    print("\nResults:")
    print(df[['url', 'avg_sentence_length', 'num_sentences', 'num_paragraphs',
              'num_words', 'sneaker_price', 'article_type', 'sneaker_brand']])



if __name__ == "__main__":
    main()
