import pandas as pd
import numpy as np
import re
import nltk
import kagglehub
import os

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

path = kagglehub.dataset_download("hijest/genre-classification-dataset-imdb")
print("Path to dataset files:", path)

DATA_PATH = os.path.join(path, "Genre Classification Dataset", "train_data.txt")

df = pd.read_csv(
    DATA_PATH,
    sep=":::",
    engine="python",
    names=["id", "title", "genre", "description"]
)

df = df[['description', 'genre']].dropna()

print("Dataset Shape:", df.shape)
print(df.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)

df['clean_description'] = df['description'].apply(clean_text)

X = df['clean_description']
y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Feature Shape:", X_train_tfidf.shape)

def evaluate_model(model, name):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print("\n" + "=" * 50)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 50)

nb_model = MultinomialNB()
evaluate_model(nb_model, "Multinomial Naive Bayes")

lr_model = LogisticRegression(max_iter=300, n_jobs=-1)
evaluate_model(lr_model, "Logistic Regression")

svm_model = LinearSVC()
evaluate_model(svm_model, "Linear SVM")

def predict_genre(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    return svm_model.predict(vector)[0]

sample_text = "A group of astronauts travel through space to save humanity"
print("\nSample Prediction:", predict_genre(sample_text))
