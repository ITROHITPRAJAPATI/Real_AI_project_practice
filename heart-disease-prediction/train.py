import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_data():
    return pd.read_csv("data/heart.csv")

def train_model(df):
    X=df[["Age","BP","Cholesterol"]]
    y=df["HeartDisease"]
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )
    model=DecisionTreeClassifier(random_state=42)
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    print("Prediction :",prediction)
    print("Actual :",y_test.values)
    return model

def save_model(model):
    joblib.dump(model,"models/heart_model.pkl")
    print("Model Saved Successfully.")

def main():
    df=load_data()
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.isnull().sum())
    model=train_model(df)
    save_model(model)

if __name__=="__main__":
    main()