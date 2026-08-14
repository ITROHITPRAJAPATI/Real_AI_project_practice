import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def load_data():
    df=pd.read_csv("data/titanic.csv")
    return df

def train_model(df):

    #select required columns
    df=df[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare"]]

    #Handal missing Values
    df["Age"]=df["Age"].fillna(df["Age"].median())
    df["Fare"]=df["Fare"].fillna(df["Fare"].median())

    #Convert Sex into numeric
    encoder=LabelEncoder()
    df["Sex"]=encoder.fit_transform(df["Sex"])

    #Feature and target
    X=df.drop("Survived",axis=1)
    y=df["Survived"]
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)

    #Check Accuracy
    accuracy=model.score(X_test,y_test)
    print("Model Accuracy :",accuracy)
    return model,encoder

def save_model(model,encoder):
    joblib.dump(
        {
            "model":model,
            "encoder":encoder
        },
        "models/titanic_model.pkl"
    )
    print("Model Sved Successfully.")

def main():
    try:
        df=load_data()
        model,encoder=train_model(df)
        save_model(model,encoder)
    except Exception as e:
        print("Training Error",e)

if __name__=="__main__":
    main()