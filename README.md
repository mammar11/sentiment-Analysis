# Sentiment Analysis of Product Reviews

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Sentiment Analysis](https://img.shields.io/badge/Sentiment-Analysis-brightgreen.svg)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data Collection](#data-collection)
- [Preprocessing](#preprocessing)
- [Sentiment Analysis](#sentiment-analysis)
- [Results](#results)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project performs sentiment analysis on product reviews collected from Walmart and Amazon. It aims to classify customer feedback as positive, neutral, or negative by applying various natural language processing (NLP) techniques and models.

## Project Overview

Key steps in this project include:

1. **Data Collection**: Scraping product reviews from Walmart and Amazon and storing them in CSV files.
2. **Data Preprocessing**: Cleaning and preparing the text data for analysis through tokenization and part-of-speech tagging.
3. **Sentiment Analysis**:
   - Using traditional tools like **NLTK** and **VADER** for basic sentiment analysis.
   - Utilizing **RoBERTa**, a transformer-based model, for advanced sentiment classification.
   - Visualizing sentiment trends and patterns with **SVGling**.
4. **Performance Evaluation**: Comparing results and analyzing model accuracy and insights.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - **Scraping**: BeautifulSoup, Requests
  - **NLP**: NLTK, VADER, RoBERTa, TQDM
  - **Visualization**: Matplotlib, SVGling
  - **Data Handling**: Pandas, CSV

## Data Collection

The dataset consists of product reviews scraped from **Walmart** and **Amazon**. The reviews are stored in separate CSV files with columns such as `review_text`, `rating`, and `source` (indicating the platform).

- **Scraping Method**: BeautifulSoup was used to parse HTML data, and the data is stored in `walmart_reviews.csv` and `amazon_reviews.csv`.

## Preprocessing

Before conducting sentiment analysis, the text data is cleaned and preprocessed through these steps:

- **Tokenization**: Breaking down sentences into words or tokens for further analysis.
- **Part-of-Speech Tagging**: Labeling words based on their grammatical function to support sentiment interpretation.
- **Data Cleaning**: Removing irrelevant information like special characters, punctuation, and numbers.

### Code Snippet

```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load data
data = pd.read_csv('walmart_reviews.csv')

# Tokenization
data['tokens'] = data['review_text'].apply(word_tokenize)

# Stopword Removal
stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
