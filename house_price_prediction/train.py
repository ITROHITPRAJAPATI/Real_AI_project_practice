import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
df=pd.read_csv("data/house_price.csv")
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())
X=df.drop("Price",axis=1)
y=df["Price"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42
)
model=LinearRegression()
model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# print("Prediction :",y_pred)
# print("Actual :",y_test.values)
joblib.dump(model,"models/house_price_model.pkl")
print("Model Save Successfully")