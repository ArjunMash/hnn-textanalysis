import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import re
from typing import List, Literal, Optional
from pydantic import BaseModel
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import json

# Handle requests from a script run and from Streamlit run
try:
    from prompts.prompt1 import SYSTEM_PROMPT # Runing this script
except ModuleNotFoundError:
    from .prompts.prompt1 import SYSTEM_PROMPT # Streamlit app

# Load environment variables from .env file
load_dotenv()

# Define Structured Output for OpenAI's return:
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

# A checkpoint for number of rows to save progress at
CHECKPOINT_INTERVAL = 50


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
    """Uses OpenAI API and structured outputs to return the article type, primary sneaker brand and sneaker price. Original sync function"""
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

# Sync function to generate embeddings using OpenAI text-embedding-3-small
def get_embedding(text):
    """Generate embedding using OpenAI text-embedding-3-small"""
    # Clean the text
    text = text.replace("\n", " ").strip()

    # Generate embedding
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )

    return response.data[0].embedding


# Used Claude to help implement these features to allow parallelization while not exceeding OpenAI RPM/TPM limits
class RateLimiter:
    """Tracks rate limits from OpenAI response headers and calculates adaptive delays"""
    def __init__(self):
        self.remaining_requests = float('inf')
        self.limit_requests = float('inf')
        self.remaining_tokens = float('inf')
        self.limit_tokens = float('inf')

    def update_from_headers(self, headers):
        """Update rate limit state from response headers"""
        self.remaining_requests = int(headers.get('x-ratelimit-remaining-requests', self.remaining_requests))
        self.limit_requests = int(headers.get('x-ratelimit-limit-requests', self.limit_requests))
        self.remaining_tokens = int(headers.get('x-ratelimit-remaining-tokens', self.remaining_tokens))
        self.limit_tokens = int(headers.get('x-ratelimit-limit-tokens', self.limit_tokens))

    async def get_delay(self) -> float:
        """Calculate delay based on remaining capacity"""
        if self.remaining_requests == float('inf'):
            return 0.1

        utilization = 1 - (self.remaining_requests / self.limit_requests)

        if utilization > 0.9:
            return 2.0  # Aggressive backoff
        elif utilization > 0.8:
            return 0.5  # Moderate backoff
        else:
            return 0.1  # Normal pacing



async def get_embeddings_async(async_client, body):
    """Generate embeddings using OpenAI text-embedding-3-small"""
    # Clean the text (remove newlines as recommended by OpenAI)
    text = body.replace("\n", " ")

    # Generate embedding
    response = await async_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    # Return the vector embedding
    return response.data[0].embedding


async def gpt_process_async(async_client, body, rate_limiter):
    """Async version of gpt_process with rate limiting"""
    # Make API call with raw response to get headers
    raw_response = await async_client.responses.with_raw_response.parse(
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

    # Update rate limiter from headers
    rate_limiter.update_from_headers(raw_response.headers)

    # Apply adaptive delay
    delay = await rate_limiter.get_delay()
    await asyncio.sleep(delay)

    # Parse the parsed response from the raw response
    parsed_response = raw_response.parse()
    info = parsed_response.output_parsed

    return info.sneaker_price, info.article_type, info.sneaker_brand


async def save_checkpoint(df, output_file):
    """Atomic save with temp file + rename"""
    temp_file = f"{output_file}.tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, output_file)
    print(f"✓ Checkpoint saved: {output_file}")


def load_and_resume(input_file):
    """Load CSV and identify rows needing processing"""
    df = pd.read_csv(input_file)

    # Add columns if missing
    new_columns = {
        'sneaker_price': None,
        'article_type': None,
        'sneaker_brand': None,
        'avg_sentence_length': None,
        'num_sentences': None,
        'num_paragraphs': None,
        'num_words': None,
        'text_embedding': None
    }

    for col, default in new_columns.items():
        if col not in df.columns:
            df[col] = default

    # Find unprocessed rows (where GPT features are missing)
    mask = df['sneaker_price'].isna() & df['article_type'].isna()
    pending_indices = df[mask].index.tolist()

    print(f"Loaded {len(df)} rows, {len(pending_indices)} pending processing")
    return df, pending_indices


async def process_single_row(async_client, df, row_index, semaphore, rate_limiter):
    """Process one row with semaphore control"""
    async with semaphore: # Note to self: semaphores are like a global lock to prevent multiple threads from touching the same block
        try:
            article_text = df.loc[row_index, 'text']

            # Skip rows with missing text
            if pd.isna(article_text) or not isinstance(article_text, str) or not article_text.strip():
                return {
                    'success': False,
                    'row_index': row_index,
                    'url': df.loc[row_index, 'url'] if 'url' in df.columns else 'N/A',
                    'error_type': 'MissingText',
                    'error_message': 'No article text available',
                    'timestamp': datetime.now().isoformat()
                }

            # Process body structure (sync, fast)
            sen_len, num_s, num_p, num_w = get_body_struct(article_text)

            # Process with GPT (async, slow)
            sneaker_price, article_type, sneaker_brand = await gpt_process_async(
                async_client, article_text, rate_limiter
            )

            # Generate embeddings (async, fast)
            embedding = await get_embeddings_async(async_client, article_text)

            # Update DataFrame
            df.loc[row_index, 'avg_sentence_length'] = sen_len
            df.loc[row_index, 'num_sentences'] = num_s
            df.loc[row_index, 'num_paragraphs'] = num_p
            df.loc[row_index, 'num_words'] = num_w
            df.loc[row_index, 'sneaker_price'] = sneaker_price
            df.loc[row_index, 'article_type'] = article_type
            df.loc[row_index, 'sneaker_brand'] = sneaker_brand
            df.loc[row_index, 'text_embedding'] = json.dumps(embedding)

            return {'success': True, 'row_index': row_index}

        except Exception as e:
            print(f"✗ Error processing row {row_index}: {type(e).__name__}: {e}")
            # Get URL for error log
            url = df.loc[row_index, 'url'] if 'url' in df.columns else 'N/A'
            return {
                'success': False,
                'row_index': row_index,
                'url': url,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }


async def process_batch_async(
    input_file: str,
    output_file: str,
    max_concurrent: int = 50,
    checkpoint_interval: int = 50,
    max_rows: int = None
):
    """ Orchestrator for the parallel processing of feature enhancement"""

    # Load data and find pending rows
    df, pending_indices = load_and_resume(input_file)

    if max_rows:
        pending_indices = pending_indices[:max_rows]

    if not pending_indices:
        print("No rows to process!")
        return

    print(f"Processing {len(pending_indices)} rows with {max_concurrent} concurrent requests")

    # Initialize async client and rate limiter
    async with AsyncOpenAI(api_key=OPENAI_API_KEY) as async_client:
        semaphore = asyncio.Semaphore(max_concurrent)
        rate_limiter = RateLimiter()

        # Create tasks
        tasks = [
            process_single_row(async_client, df, idx, semaphore, rate_limiter)
            for idx in pending_indices
        ]

        # Process with progress tracking
        start_time = datetime.now()
        processed = 0
        errors = []

        for coro in asyncio.as_completed(tasks):
            result = await coro
            processed += 1

            if not result['success']:
                errors.append(result)

            # Progress update
            if processed % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed
                remaining = len(pending_indices) - processed
                eta_seconds = remaining / rate if rate > 0 else 0

                print(f"Progress: {processed}/{len(pending_indices)} "
                      f"({processed/len(pending_indices)*100:.1f}%) | "
                      f"Rate: {rate:.2f} rows/sec | "
                      f"ETA: {int(eta_seconds/60)}m {int(eta_seconds%60)}s")

            # Checkpoint trigger
            if processed % checkpoint_interval == 0:
                await save_checkpoint(df, output_file)

        # Final save
        await save_checkpoint(df, output_file)

        # Save error log if there were errors
        if errors:
            error_df = pd.DataFrame(errors)
            error_log_file = 'data/gpt_processing_errors.csv'
            error_df.to_csv(error_log_file, index=False)
            print(f"\n✓ Error log saved: {error_log_file}")

        # Summary
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total: {processed} rows")
        print(f"Success: {processed - len(errors)} rows")
        print(f"Errors: {len(errors)} rows")
        print(f"Time: {(datetime.now() - start_time).total_seconds():.1f}s")
        print(f"Rate: {processed / (datetime.now() - start_time).total_seconds():.2f} rows/sec")
        print(f"{'='*60}")

        if errors:
            print(f"\n⚠ {len(errors)} rows failed. Rerun script to retry (auto-detects unprocessed rows).")


async def main_parallel(
    input_file: str = 'data/hnhh_enriched.csv',
    output_file: str = 'data/hnhh_processed.csv',
    max_concurrent: int = 50,
    checkpoint_interval: int = 50,
    max_rows: int = None
):
    """Async entry point for parallel batch processing"""

    print(f"{'='*60}")
    print(f"OpenAI GPT Parallel Batch Processing")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Concurrency: {max_concurrent}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print(f"{'='*60}\n")

    await process_batch_async(
        input_file=input_file,
        output_file=output_file,
        max_concurrent=max_concurrent,
        checkpoint_interval=checkpoint_interval,
        max_rows=max_rows
    )


def main():
    df = pd.read_csv('data/hnhh_enriched.csv', nrows=10)
    print("Running Test Script w/ Sync Functions")

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

        # Generate embeddings:
        df.loc[i, 'text_embedding'] = get_embedding(article_text)
    # Display results
    print("\nResults:")
    print(df[['url', 'avg_sentence_length', 'num_sentences', 'num_paragraphs',
              'num_words', 'sneaker_price', 'article_type', 'sneaker_brand']])



if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "parallel":
        # Usage: python feature_eng.py parallel [max_rows] [concurrency]
        max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else None
        concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 50

        asyncio.run(main_parallel(
            input_file='data/hnhh_enriched.csv',
            output_file='data/hnhh_processed.csv',
            max_concurrent=concurrency,
            max_rows=max_rows
        ))
    else:
        # Sync version for testing
        main()