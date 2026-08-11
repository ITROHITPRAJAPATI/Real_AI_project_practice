import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH
from utils import save_model
from logger import logger

def load_data():
    return pd.read_csv(DATA_PATH)

def train_model(df):
    X=df[["Age","Salary"]]
    y=df["Purchased"]
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,
        random_state=42
    )
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled,y_train)
    accuracy=model.score(X_test_scaled,y_test)
    print("model accuracy",accuracy)
    return model,scaler

def main():
    try:
        df=load_data()
        model,scaler=train_model(df)
        save_model(model,scaler)
        logger.info("KNN Model Training Successfully")
        print("Model Saved Successfully.")
    except Exception as e:
        logger.error(f"training error {e}")
        print("Traing Failed",e)

if __name__=="__main__":
    main()
    