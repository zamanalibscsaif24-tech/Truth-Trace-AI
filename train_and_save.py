import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import os

print("Loading cleaned dataset TruthTrace_AI.csv...")
try:
    df = pd.read_csv("TruthTrace_AI.csv", encoding='utf-8')
except FileNotFoundError:
    print("TruthTrace_AI.csv not found. Please ensure it is in the same directory.")
    exit()

print("Dataset loaded. Shape:", df.shape)

# Drop any potential NaNs in combined_text
df = df.dropna(subset=['combined_text'])

# Split target and features (We will use the whole dataset to train for the final model to maximize data)
X_text = df['combined_text'].astype(str)
y = df['target']

# 1. Vectorize the text
print("Fitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X_text)

# 2. Train the Model
print("Training Logistic Regression Model on TF-IDF features only...")
# Using typical strong parameters for this type of problem
model = LogisticRegression(C=10, max_iter=1000)
model.fit(X_tfidf, y)

# 5. Save the models
print("Saving models to disk...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(model, 'logreg_model.pkl')

print("Models saved successfully as 'tfidf_vectorizer.pkl' and 'logreg_model.pkl'.")
