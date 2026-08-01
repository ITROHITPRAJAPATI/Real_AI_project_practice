import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Load Dataset
def load_data():
    return pd.read_csv("data/loan.csv")

#Train Model
def train_model(df):
    X=df[["Age","Income","CreditScore"]]
    y=df["LoanApproved"]
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )
    model=LogisticRegression()
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    print("Predictin",prediction)
    print("Actual :",y_test.values)
    return model

#Save Model
def save_model(model):
    joblib.dump(model,"models/loan_model.pkl")
    print("Model Save Successfully")

# Main Function
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