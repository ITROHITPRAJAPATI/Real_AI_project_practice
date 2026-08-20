import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load Data
df=pd.read_csv("data/employee_attrition.csv")
# Convert caterical values into numbers
df["OverTime"]=df["OverTime"].map({"Yes":1,"No":0})
df["Attrition"]=df["Attrition"].map({"Yes":1,"No":0})
# Select feature and target
X=df[["Age","JobSatisfaction","MonthlyIncome","YearsAtCompany","OverTime","JobLevel"]]
y=df["Attrition"]
#Split data
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42,
    stratify=y
)
# creat model
model=DecisionTreeClassifier(random_state=42,max_depth=4)
# Train model
model.fit(X_train,y_train)
# Make Prediction
y_pred=model.predict(X_test)
# Chek Accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy :",accuracy)
#Save Model
joblib.dump(model,"models/employee_attrition_model.pkl")
print("Model Saved Successfully.")