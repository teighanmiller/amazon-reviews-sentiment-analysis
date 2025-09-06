# Amazon Reviews Sentiment Analysis  

This project demonstrates a **machine learning pipeline** for classifying Amazon product reviews into **positive** and **negative sentiment**. It showcases my ability to build **end-to-end NLP systems** that handle data ingestion, preprocessing, model training, evaluation, and persistence.  

---

## Project Overview  

E-commerce platforms like Amazon generate massive amounts of customer reviews. Analyzing this feedback manually is impossible at scale. This project automates the process using **logistic regression with bag-of-words features**, providing an interpretable baseline model for sentiment classification.  

The pipeline covers:  
- **Preprocessing** raw text with spaCy (tokenization, lemmatization, stopword & punctuation removal).  
- **Feature engineering** with scikit-learn’s `CountVectorizer` and `TfidfVectorizer`.  
- **Model training & persistence** with logistic regression.  
- **Evaluation** using accuracy, precision, recall, and F1-score.  

---

## Skills & Tools Demonstrated  

- **Natural Language Processing (NLP)**: spaCy for text cleaning and efficient batch processing.  
- **Machine Learning**: Logistic regression as a baseline classifier for sentiment analysis.  
- **Feature Extraction**: Compared bag-of-words (`CountVectorizer`) vs. TF-IDF (`TfidfVectorizer`).  
- **Software Engineering Practices**:  
  - Modular class design (`Preprocessing` and `SentimentAnalysis`).  
  - Serialization of models with `pickle`.  
  - Docstrings, comments, and type hints for maintainability.  
- **Data Handling**: Efficient file parsing and label encoding.  

---

## Results  

- Using **CountVectorizer + Logistic Regression**:  
  - **Accuracy**: ~0.888
  - Strong balance of precision and recall across both classes.  

- Using **TF-IDF + Logistic Regression**:  
  - **Accuracy**: ~0.887
  - Slightly lower recall on the positive class.

**Why CountVectorizer performed slightly better**:  
Amazon reviews often include repeated, high-frequency words that strongly indicate sentiment (e.g., *“good,” “bad,” “terrible,” “amazing”*). TF-IDF slightly down-weights these frequent words, which caused a marginal drop in performance. CountVectorizer retains their full influence, giving a tiny edge in this dataset.  

---

## 🚀 How It Works  

1. **Training**  
   - Preprocess training data (`train.ft.txt`).  
   - Fit vectorizer (`CountVectorizer` or `TfidfVectorizer`).  
   - Train logistic regression model.  
   - Save artifacts (`vectorizer.pkl`, `sentiment_model.pkl`).  

2. **Testing / Prediction**  
   - Preprocess test data (`test.ft.txt`).  
   - Load saved model artifacts.  
   - Predict and evaluate sentiment classification.  

---

## Project Structure
- amazon-reviews-sentiment-analysis/
  - data/                      # Training & test data, model artifacts
  - src/
    - preprocessing.py         # Text preprocessing pipeline
    - sentiment_analysis.py    # Model training, saving, loading, prediction
    - sentiment_module.py      # Orchestrates training & evaluation
  - requirements.txt           # Python dependencies

## Key Takeaways  

- Built an **end-to-end machine learning workflow** from scratch.  
- Applied **NLP techniques** and tested multiple feature extraction strategies.  
- Learned the importance of **choosing the right representation** (CountVectorizer outperformed TF-IDF on this dataset).  
- Designed the system to be easily extensible, including support for:  
  - `TfidfVectorizer`  
  - Deep learning models such as BERT or RoBERTa  
  - Serving via REST API frameworks like Flask or FastAPI  

---

## Next Steps  

If extended into production, potential improvements include:  
- Deploying the model as an API for real-time predictions.  
- Adding experiment tracking using tools like MLflow or Weights & Biases.  
- Incorporating deep learning embeddings for higher accuracy.  
- Expanding to multilingual sentiment classification.  

---
