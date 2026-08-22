# Fake News Detection

## Project Overview

Fake News Detection is a Machine Learning project that analyzes news text
and predicts whether the news is real or fake.

The project uses Natural Language Processing (NLP) techniques to convert
text into numerical features and a Logistic Regression model for
classification.

## Objective

The main objective of this project is to automatically classify news
articles into two categories:

- Real News
- Fake News

This project demonstrates a complete Machine Learning workflow from
data validation and preprocessing to model training and prediction.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- Git & GitHub

## Machine Learning Workflow

1. Load the news dataset
2. Validate the dataset
3. Clean and preprocess the text data
4. Split data into training and testing sets
5. Convert text into numerical features using TF-IDF
6. Train Logistic Regression model
7. Evaluate model performance
8. Save the trained model and vectorizer
9. Predict whether new news is real or fake

## Project Structure

```text
fake-news-detection/
│
├── data/
│   ├── raw/
│   │   └── news.csv
│   │
│   └── processing/
│       └── clean_news.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── validate_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── logs/
│
└── README.md
## Author : Rohit Prajapati