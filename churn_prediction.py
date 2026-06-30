
import pandas as pd
import numpy as np
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = kagglehub.dataset_download(
    "shantanudhakadd/bank-customer-churn-prediction"
)

print("Path to dataset files:", path)

DATA_PATH = os.path.join(path, "Churn_Modelling.csv")

df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
print(df.head())


df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Geography"] = label_encoder.fit_transform(df["Geography"])


X = df.drop("Exited", axis=1)
y = df["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 55)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("=" * 55)




nb_model = GaussianNB()
evaluate_model(nb_model, "Gaussian Naive Bayes")


lr_model = LogisticRegression(max_iter=500)
evaluate_model(lr_model, "Logistic Regression")

svm_model = SVC(kernel="rbf", probability=True)
evaluate_model(svm_model, "Support Vector Machine")

def predict_churn(sample_data):
    """
    sample_data format:
    [CreditScore, Geography, Gender, Age, Tenure, Balance,
     NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
    """
    sample_data = scaler.transform([sample_data])
    prediction = lr_model.predict(sample_data)[0]
    return "Churn" if prediction == 1 else "Not Churn"

# Example customer
sample_customer = [
    600,  # CreditScore
    1,    # Geography (encoded)
    0,    # Gender (encoded)
    40,   # Age
    5,    # Tenure
    60000, # Balance
    2,    # NumOfProducts
    1,    # HasCrCard
    1,    # IsActiveMember
    50000 # EstimatedSalary
]

print("\nSample Prediction:", predict_churn(sample_customer))
