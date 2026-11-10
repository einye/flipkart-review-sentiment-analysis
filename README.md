# Flipkart Product Review Sentiment Analysis

Sentiment analysis on Flipkart product reviews using two complementary approaches: **VADER** (lexicon-based) and a **stacked LSTM (RNN)** deep learning model. This project explores rating-based vs. text-based sentiment, visualizes trends across products and star ratings, and highlights the challenges of classifying neutral sentiment.

## Overview

Customer reviews are a goldmine for product feedback, but star ratings alone don't tell the full story. This project investigates:

- How well do written reviews align with star ratings?
- Can a lightweight RNN classify sentiment into **Positive / Neutral / Negative**?
- Where does lexicon-based sentiment (VADER) fall short (e.g., sarcasm, ambiguity)?

## Dataset

- **Source:** `flipkart_product.csv`
- **Columns:** `ProductName`, `Price`, `Rate`, `Review`, `Summary`
- **Encoding:** `ISO/IEC 8859-1 (Latin-1)`

## Preprocessing

- Dropped rows with missing `Summary`; filled missing `Review` with placeholder
- Cleaned `Price` using regex (removed `₹`, commas, and non-numeric characters)
- Converted `Rate` to integer, dropped invalid entries
- Removed non-alphanumeric characters from `ProductName`
- Lowercased and stripped punctuation from `Review` / `Summary` text

## Approach

### 1. Rating-Based Sentiment

Mapped star ratings to sentiment:
| Rating | Sentiment |
|--------|-----------|
| 4–5  | Positive |
| 3 | Neutral  |
| 1–2 | Negative |

### 2. Text-Based Sentiment (VADER)

Used NLTK's `SentimentIntensityAnalyzer` on the `Summary` column with standard thresholds (±0.05).

### 3. RNN Model (LSTM)

- Tokenized `Summary` (vocab = 10,000, max length = 100)
- Embedding (100-d) → LSTM (64 units) → Dropout (0.2) → Dense (softmax, 3 classes)
- Trained 5 epochs, batch size 32, 80/20 train-validation split
- Evaluated on 20% hold-out test set

## Results

### Agreement Between Rating & Text Sentiment
**82.70%** - suggesting strong overall alignment, with notable exceptions at lower ratings (likely sarcasm or mixed feedback).

### RNN Model Performance

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Negative | 0.86      | 0.85   | 0.85     |
| Neutral  | 0.44      | 0.21   | 0.29     |
| Positive | 0.93      | 0.98   | 0.95     |
| **Overall Accuracy** | | | **0.90** |

**Key takeaway:** The model performs well on Positive and Negative classes but struggles with Neutral reviews due to class imbalance and the ambiguity of neutral language.

## Visualizations

The notebook includes:
- Sentiment distribution (text-based vs. rating-based)
- Sentiment breakdown by star rating
- Sentiment across top 10 products by review count
- Word clouds for **Positive**, **Negative**, and **Neutral** reviews

## Key Findings

1. **Positive bias** - ~78% of reviews (both text and rating-based) are positive, possibly reflecting selection bias or promotional reviews.
2. **Neutral class is underrepresented and hard to detect** - F1-score of 0.29 suggests the LSTM needs class-balancing (oversampling, class weights) to improve.
3. **Rating vs. text mismatch at 1–2 stars** - Some low-rated reviews have positive/neutral text, likely due to sarcasm or VADER's limited handling of Hinglish/slang.
4. **VADER is fast but shallow** - contextual models (e.g., BERT) would likely improve neutral and sarcastic review detection.

## Acknowledgements

**This project is for educational and analytical purposes. Dataset ownership belongs to Flipkart and the original Kaggle uploader, Mansi Thummar. [Link](https://www.kaggle.com/datasets/mansithummar67/flipkart-product-review-dataset/data)**