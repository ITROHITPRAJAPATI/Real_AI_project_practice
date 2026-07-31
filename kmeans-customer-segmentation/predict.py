import joblib
import pandas as pd
model=joblib.load("models/kmeans_model.pkl")
try:
    age=int(input("Enter Age :"))
    salary=int(input("Enter Salary :"))
    if age<18:
        print("you age less then 18 ")
        exit()
    new_customer=pd.DataFrame({"Age":[age],"Salary":[salary]})
    prediction=model.predict(new_customer)
    print("\n----- Prediction Result -----")
    print("Age :",age)
    print("Salary :",salary)
    print("Prediction Cluster :",prediction[0])
except ValueError:
    print("Pleas enter numeric values")
