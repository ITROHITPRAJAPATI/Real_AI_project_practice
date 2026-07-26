import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
df=pd.read_csv("loan.csv")
# print(df.head())
# print(df.shape)
# print(df.columns)
x=df[["Income","Age","Credit_Score"]]
y=df["Loan"]
# print("\n Feature (x) :",x)
# print("\n Target (y):",y)
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,
    random_state=42
)
# print("x_train Shape",x_train.shape)
# print("x_test Shape",x_test.shape)
# print("y_train Shape",y_train.shape)
# print("y_test Shape",y_test.shape)
model=LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(model,"loan_model.pkl")
# y_pred=model.predict(x_test)
# print("Prediction",y_pred)
# accuracy=accuracy_score(y_test,y_pred)
# print("Accuracy",accuracy)
loaded_model=joblib.load("loan_model.pkl")
def get_input():
    try:
        income=int(input("Enter Income : "))
        age=int(input("Enter Age : "))
        credit_score=int(input("Enter Credit Score : "))
        if income<=0 or age<=0 or credit_score<=0:
            print("Invalid Inter")
            return None
        return income,age,credit_score
    except ValueError:
        print("Pleas enter valid numeric value")
        return None
def main():
    data=get_input()
    if data is None:
        return
    income,age,credit_score=data
    new_data=pd.DataFrame({"Income":[income],"Age":[age],"Credit_Score":[credit_score]})
    result=loaded_model.predict(new_data)
    print("\n----- Loan Approvel Prediction -----")
    if result[0]=="Yes":
        print("Congratulation Our Loan Approvel Pass ")
    else:
        print("Sorry not Loan Approvel ")

main()