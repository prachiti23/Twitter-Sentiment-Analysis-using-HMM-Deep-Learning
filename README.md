# Twitter-Sentiment-Analysis-using-HMM-Deep-Learning

## Project Overview

This project performs Twitter sentiment analysis on movie reviews using Hidden Markov Model (HMM), traditional Machine Learning (ML), and Deep Learning (DL) techniques.

Social media platforms like Twitter serve as valuable sources of public opinion, where users express their sentiments about various topics, including movies. Sentiment Analysis helps in classifying these opinions into positive or negative sentiments, which is useful for:

Movie review analysis to understand audience reception.

Brand reputation management for companies monitoring social media.

Market research for predicting product or service popularity.

##  Objectives

✅ Collect and preprocess Twitter movie reviews.

✅ Implement different ML & DL algorithms for sentiment classification.

✅ Compare performance metrics (accuracy, precision, recall, F1-score).

✅ Identify the best-performing model for sentiment analysis.

## Technologies & Libraries Used

The project is implemented using Python with the following key libraries:

Data Processing : pandas, numpy

NLP Techniques : nltk, re, string, wordcloud

Machine Learning : scikit-learn (Naïve Bayes, Logistic Regression, SVM, KNN)

Deep Learning : tensorflow, keras (LSTM, BiLSTM, GRU, RNN)

Data Visualization : matplotlib, seaborn

## Dataset Details

The dataset for this project is obtained from Kaggle's Movie Review Dataset, containing:

* Tweet text (user-generated movie reviews)

* Sentiment labels:

  1 → Positive sentiment

  0 → Negative sentiment

## Preprocessing Steps:

✅ Text Cleaning: Remove special characters, URLs, punctuation, and stopwords.

✅ Tokenization & Lemmatization: Convert text into meaningful tokens.

✅ Vectorization: Convert text data into numerical format using TF-IDF or Word2Vec.

## Algorithms Implemented

We compare multiple ML & DL models for sentiment classification:

#### 🔹 Machine Learning Models

✔️ Naïve Bayes – A probabilistic model best suited for text classification.

✔️ Logistic Regression – A linear model that predicts the probability of sentiment.

✔️ SVM (Support Vector Machine) – Finds the best hyperplane to classify text data.

✔️ KNN (K-Nearest Neighbors) – Classifies based on similarity with nearest tweets.

#### 🔹 Deep Learning Models

✔️ LSTM (Long Short-Term Memory) – Specialized for sequence data like text.

✔️ BiLSTM (Bidirectional LSTM) – Captures both forward and backward dependencies.

✔️ GRU (Gated Recurrent Unit) – A simplified version of LSTM.
            
✔️ RNN (Recurrent Neural Network) – Processes sequential text data.

 ## Model Performance Comparison

After training, we evaluate models using accuracy, precision, recall, and F1-score.

Naïve Bayes : XX%

Logistic Regression : XX%

SVM : 94.78%

KNN : 95.47%

LSTM : 99.87%

BiLSTM : 82.56%

GRU : 84.74%

RNN : 78.69%

## Key Insights:

✔️ Deep learning models outperformed traditional ML models.

✔️ BiLSTM achieved the highest accuracy, followed by LSTM and GRU.

## System Architecture

The following architecture is used for the sentiment analysis pipeline:

🔹 Hidden Markov Model (HMM) for Sentiment Analysis

HMM is implemented to capture the probability distribution of word sequences in positive and negative sentiment categories.

## Installation & Usage

1️⃣ Install Required Libraries

Run the following command to install dependencies:

2️⃣ Clone the Repository

3️⃣ Run Jupyter Notebook

Open FINAL_PROJECT_SENTIMENTAL_ANALYSIS_DEEP_LEARNING.ipynb to explore the code.

4️⃣ Run Python Script (Optional)

If you have a .py version of the notebook:

## Visualization & Insights

We use matplotlib and seaborn for:

✅ Sentiment distribution visualization.

✅ Wordclouds for positive and negative words.

✅ Model performance comparison plots.

Example Wordcloud for Positive & Negative Tweets:


## Future Improvements

🔹 Implement transformer-based models like BERT, RoBERTa for better accuracy.

🔹 Optimize hyperparameters using GridSearchCV for improved performance.

🔹 Expand dataset with real-time Twitter API integration.
