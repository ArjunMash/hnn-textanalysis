import streamlit as st
import torch
import numpy as np
import pandas as pd
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add scripts directory to path to import model
from scripts.models.model1 import SimpleRegressorNet, load_model
from scripts.models.feature_eng import get_body_struct, get_embedding

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# App title
st.title("Article Pageview Predictor")
st.write("Predict how many pageviews your sneaker article might get")

# Load model (cache it so it's only loaded once)
@st.cache_resource
def load_trained_model():
    """Load the trained model, scaler, and category info"""
    model_path = "scripts/models/pageviews_model.pt"
    try:
        model, scaler, category_info = load_model(model_path)
        return model, scaler, category_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, category_info = load_trained_model()

if model is None:
    st.error("Failed to load model. Please ensure the model file exists at scripts/models/pageviews_model.pt")
    st.stop()

def prepare_input_features(text, article_type, sneaker_brand, sneaker_price, publish_date, category_info):
    """Prepare input features matching training format"""

    # Get embedding
    with st.spinner("Generating text embedding..."):
        embedding = get_embedding(text)
    embedding_array = np.array(embedding)

    # Extract text features using get_body_struct from feature_eng
    avg_sentence_length, num_sentences, num_paragraphs, num_words = get_body_struct(text)
    text_features = {
        'avg_sentence_length': avg_sentence_length,
        'num_sentences': num_sentences,
        'num_paragraphs': num_paragraphs,
        'num_words': num_words
    }

    # Date features
    publish_datetime = pd.to_datetime(publish_date)
    day_of_week = publish_datetime.dayofweek  # 0 for Mon, 6 for Sun
    month = publish_datetime.month

    # Handle sneaker price
    has_sneaker_price = 1 if sneaker_price > 0 else 0
    sneaker_price_value = sneaker_price if sneaker_price > 0 else 0

    # Casting Numeric features (Matching training order)
    numeric_features = np.array([
        text_features['avg_sentence_length'],
        text_features['num_sentences'],
        text_features['num_paragraphs'],
        text_features['num_words'],
        sneaker_price_value,
        has_sneaker_price,
        day_of_week,
        month
    ])

    # Categorical features - one-hot encode using exact columns from training
    article_type_encoded = np.array([
        1 if f'article_{article_type}' == col else 0
        for col in category_info['article_types']
    ])

    brand_encoded = np.array([
        1 if f'brand_{sneaker_brand}' == col else 0
        for col in category_info['brands']
    ])

    # Concatenate all features
    X = np.concatenate([embedding_array, numeric_features, article_type_encoded, brand_encoded])

    return X.reshape(1, -1)  # Reshape for single prediction

# Sidebar for inputs
st.sidebar.header("Article Details")

# Text input
article_text = st.text_area(
    "Article Text",
    height=300,
    placeholder="Paste your article text here...",
    help="Enter the full article text. Typical articles are around 300 words."
)

# Extract available options from category_info
article_type_options = [col.replace('article_', '') for col in category_info['article_types']]
brand_options = [col.replace('brand_', '') for col in category_info['brands']]

# Article type
article_type = st.sidebar.selectbox(
    "Article Type",
    options=article_type_options,
    help="Select the type of article"
)

# Sneaker brand
sneaker_brand = st.sidebar.selectbox(
    "Sneaker Brand",
    options=brand_options,
    help="Select the primary sneaker brand featured in the article"
)

# Sneaker price
sneaker_price = st.sidebar.number_input(
    "Sneaker Price ($)",
    min_value=0.0,
    max_value=1000000.0,
    value=0.0,
    step=10.0,
    help="Enter the sneaker retail price (leave as 0 if not applicable)"
)

# Publish date
publish_date = st.sidebar.date_input(
    "Publish Date",
    value=datetime.now(),
    help="Select the publish date for the article"
)

# Predict button
if st.button("Predict Pageviews", type="primary"):
    if not article_text.strip():
        st.error("Please enter article text")
    else:
        try:
            # Prepare features
            with st.spinner("Processing article..."):
                X = prepare_input_features(
                    article_text,
                    article_type,
                    sneaker_brand,
                    sneaker_price,
                    publish_date,
                    category_info
                )

            # Scale features
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)

            # Make prediction
            model.eval()
            with torch.no_grad():
                prediction_log = model(X_tensor)

            # Convert from log space to original scale
            prediction = np.expm1(prediction_log.numpy())[0][0]

            # Display results
            st.success("Prediction Complete!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Pageviews", f"{int(prediction):,}")

            with col2:
                avg_sentence_length, num_sentences, num_paragraphs, num_words = get_body_struct(article_text)
                st.metric("Word Count", f"{num_words:,}")

            with col3:
                st.metric("Sentences", f"{num_sentences}")

            # Show article summary
            st.subheader("Article Summary")
            st.write(f"**Type:** {article_type}")
            st.write(f"**Brand:** {sneaker_brand}")
            if sneaker_price > 0:
                st.write(f"**Price:** ${sneaker_price:,.2f}")
            st.write(f"**Publish Date:** {publish_date.strftime('%B %d, %Y')}")
            st.write(f"**Paragraphs:** {num_paragraphs}")
            st.write(f"**Average Sentence Length:** {avg_sentence_length:.1f} words")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Please check that all inputs are correct and try again.")

# Footer with info
st.sidebar.markdown("---")
st.sidebar.info(
    "This model predicts article pageviews based on:\n"
    "- Article text embeddings\n"
    "- Text features (length, structure)\n"
    "- Article metadata (type, brand, price)\n"
    "- Publish timing (day, month)"
)
