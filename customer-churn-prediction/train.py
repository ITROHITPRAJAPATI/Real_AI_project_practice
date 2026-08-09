import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from config import DATA_PATH
from utils import save_model
from logger import logger

def load_data():
    return pd.read_csv(DATA_PATH)

def train_model(df):
    X=df[["Age","MonthlyCharges","Tenure"]]
    y=df["Churn"]
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )
    model=LogisticRegression()
    model.fit(X_train,y_train)
    accuracy=model.score(X_test,y_test)
    print("Model Accuracy :",accuracy)
    return model

def main():
    try:
        df=load_data()
        model=train_model(df)
        save_model(model)
        logger.info("Customer Churn Model Trained Successfully")
        print("Model Saved Successfully.")
    except Exception as e:
        logger.error(f"Training Error :{e}")
        print("Error :",e)

if __name__=="__main__":
    main()