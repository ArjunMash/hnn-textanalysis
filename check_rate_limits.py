#!/usr/bin/env python3
"""
Claude code generated script to check OpenAI API rate limits. So we don't get limited during our feature enrichment
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def check_rate_limits():
    """Make a minimal API call and display rate limit information."""

    # Initialize client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    try:
        # Make a minimal API call to check rate limits
        response = client.chat.completions.with_raw_response.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )

        # Extract rate limit headers
        headers = response.headers

        print("=" * 60)
        print("OpenAI API Rate Limits")
        print("=" * 60)
        print()

        # Request limits
        print("REQUEST LIMITS:")
        print(f"  Limit:     {headers.get('x-ratelimit-limit-requests', 'N/A')}")
        print(f"  Remaining: {headers.get('x-ratelimit-remaining-requests', 'N/A')}")
        reset_requests = headers.get('x-ratelimit-reset-requests')
        if reset_requests:
            print(f"  Resets in: {reset_requests}")
        print()

        # Token limits
        print("TOKEN LIMITS:")
        print(f"  Limit:     {headers.get('x-ratelimit-limit-tokens', 'N/A')}")
        print(f"  Remaining: {headers.get('x-ratelimit-remaining-tokens', 'N/A')}")
        reset_tokens = headers.get('x-ratelimit-reset-tokens')
        if reset_tokens:
            print(f"  Resets in: {reset_tokens}")
        print()

        print("=" * 60)
        print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    except Exception as e:
        print(f"Error checking rate limits: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(check_rate_limits())
