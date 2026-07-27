# Loan Approvel Prediction using Decision Tree
import pandas as pd 
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("loan.csv")
#print(df.head())
x=df[["Income","Age","Credit_Score"]]
y=df["Loan"]
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,
    random_state=42
)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# print(prediction)
# accuracy=accuracy_score(y_test,prediction)
# print("Accuracy : ",accuracy)
joblib.dump(model,"decision_tree_model.pkl")
loaded_model=joblib.load("decision_tree_model.pkl")
def get_input():
    try:
        income=int(input("Enter Income : "))
        age=int(input("Enter Age : "))
        credit_score=int(input("Enter Credit Score : "))
        if income<0 or age<0 or credit_score<0:
            return None
        return income,age,credit_score
    except ValueError:
        print("Pleas enter valid number ")
        return None

def predict_loan(model,income,age,credit_score):
    new_data=pd.DataFrame({"Income":[income],"Age":[age],"Credit_Score":[credit_score]})
    result=model.predict(new_data)
    if result[0]=="Yes":
        print("Loan Approved")
    else:
        print("Loan Rejected")

def main():
    data=get_input()
    if data is None:
        return
    income,age,credit_score=data
    predict_loan(loaded_model,income,age,credit_score)
    print("Thank you for using Loan Prediction System ")

main()
