import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("loan.csv")
# print(df.head())
# print(df.columns)
X=df[["Income","Age","Credit_Score"]]
y=df["Loan"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42
)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
# prediction=model.predict(X_test)
# print(prediction)
# accuracy=accuracy_score(y_test,prediction)
# print("Accuracy : ",accuracy)
joblib.dump(model,"knn_model.pkl")

#using function
def get_input():
    try:
        income=int(input("Enter Income : "))
        age=int(input("Enter Age : "))
        credit_score=int(input("Enter Credit Score : "))
        if income<0:
            print("Income can not be negative. ")
            return None
        if age<18:
            print("Age must be at least 18.")
            return None
        if credit_score<0:
            print("Credit Score con not be negative.")
            return None
        return income,age,credit_score
    except ValueError:
        print("Invalid Input! Please enter numeric values.")
        return None

def predict_loan(model,income,age,credit_score):
    new_data=pd.DataFrame({"Income":[income],"Age":[age],"Credit_Score":[credit_score]})
    prediction=model.predict(new_data)
    if prediction[0]=="Yes":
        print("Loan Approved")
    else:
        print("Loan Rejected")

def main():
    model=joblib.load("knn_model.pkl")
    data=get_input()
    if data is not None:
        income,age,credit_score=data
        predict_loan(model,income,age,credit_score)
    print("Thank You ")

if __name__=="__main__":
    main()