import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
df=pd.read_csv("data/iris.csv")
#print(df.head())
X=df.drop("target",axis=1)
y=df["target"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42
)
model=SVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Actual :",y_test)
print("Prediction :",y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy :",accuracy)
joblib.dump(model,"models/svm_model.pkl")
print("Model Saving Successfully")