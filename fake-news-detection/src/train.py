import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH="data/processed/clean_news.csv"
df=pd.read_csv(DATA_PATH)
X=df["text"]
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training Data:",len(X_train))
print("Testing Data:",len(X_test))
print("Data Completed Successfully.")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
X_train_vectorized=vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)
print("Training Feature:",X_train_vectorized.shape)
print("Testing Feature:",X_test_vectorized.shape)
print("TF-IDF feature Engineering Completed Successfully.")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
#creat Model
model=LogisticRegression(random_state=42)
#train model
model.fit(X_train_vectorized,y_train)
print("Model Training Completed Successfully.")
#prediction
y_pred=model.predict(X_test_vectorized)
#Accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print(classification_report(y_test,y_pred))
import joblib
MODEL_PATH="models/fake_news_model.pkl"
VECTORIZER_PATH="models/tfidf_vectorizer.pkl"
joblib.dump(model,MODEL_PATH)
joblib.dump(vectorizer,VECTORIZER_PATH)
print("Model and vectorizer saved Successfully.")