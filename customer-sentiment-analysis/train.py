import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# data load
df=pd.read_csv("data/reviews.csv")
#Feature and Target
X=df["review"]
y=df["sentiment"]
#convet text into numerical feature
vectorizer=TfidfVectorizer()
x_vectorized=vectorizer.fit_transform(X)
#split data
X_train,X_test,y_train,y_test=train_test_split(
    x_vectorized,y,test_size=0.2,
    random_state=42,
    stratify=y
)
#creat model
model=LogisticRegression()
#train model
model.fit(X_train,y_train)
#make Prediction
y_pred=model.predict(X_test)
#check accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Model Accuracy :",accuracy)
#Save model and vectorizer
joblib.dump(model,"model/sentiment_model.pkl")
joblib.dump(vectorizer,"model/tfidf_vectorizer.pkl")
print("Model and Vectorizer Saved Successfully.")
