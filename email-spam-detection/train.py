import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_data():
    return pd.read_csv("data/spam.csv")

def train_model(df):
    X=df["Message"]
    y=df["Spam"]
    vectorizer=CountVectorizer()
    X=vectorizer.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )
    model=MultinomialNB()
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    print("Prediction :",prediction)
    print("Actual :",y_test.values)
    return model,vectorizer

def save_model(model,vectorizer):
    joblib.dump(model,"models/spam_model.pkl")
    joblib.dump(vectorizer,"models/vectorizer.pkl")
    print("Model Saved Successfully.")

def main():
    df=load_data()
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.isnull().sum())
    model,vectorizer=train_model(df)
    save_model(model,vectorizer)

if __name__=="__main__":
    main()
