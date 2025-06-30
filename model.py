# model.py

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dataset import data

X, y = zip(*data)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

model.fit(X, y)

joblib.dump(model, "chat_model.pkl")
print("✅ Model trained and saved as chat_model.pkl")
