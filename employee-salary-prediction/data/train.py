import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Load Data
df=pd.read_csv("data/salary.csv")

#Feature $ target
X=df[["Experience"]]
y=df["Salary"]

#Split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42
)

#Train Model
model=LinearRegression()
model.fit(X_train,y_train)

#Prediction
y_pred=model.predict(X_test)
print("Prediction :",y_pred)
print("Actual :",y_test.values)

#Save Model
joblib.dump(model,"models/salary_model.pkl")
print("Model Save Successfully.")
