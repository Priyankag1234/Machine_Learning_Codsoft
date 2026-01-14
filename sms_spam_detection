
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

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Path to dataset files:", path)

DATA_PATH = os.path.join(path, "spam.csv")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1"
)

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Dataset Shape:", df.shape)
print(df.head())

df["label"] = df["label"].map({"ham": 0, "spam": 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

df["clean_message"] = df["message"].apply(clean_text)

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=30000,
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
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("=" * 50)

nb_model = MultinomialNB()
evaluate_model(nb_model, "Multinomial Naive Bayes")


lr_model = LogisticRegression(max_iter=300)
evaluate_model(lr_model, "Logistic Regression")


svm_model = LinearSVC()
evaluate_model(svm_model, "Linear SVM")


def predict_spam(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    prediction = svm_model.predict(vector)[0]
    return "SPAM" if prediction == 1 else "HAM"


sample_sms = "Congratulations! You have won a free iPhone. Click now!"
print("\nPrediction:", predict_spam(sample_sms))
