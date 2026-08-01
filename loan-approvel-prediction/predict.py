import pandas as pd
import joblib
def load_model():
    return joblib.load("models/loan_model.pkl")

def get_user_input():
    age=int(input("Enter Age :"))
    income=int(input("Enter income :"))
    credit=int(input("Enter Creadit Score :"))
    return pd.DataFrame({"Age":[age],"Income":[income],"CreditScore":[credit]})

def predict(model,new_data):
    prediction=model.predict(new_data)
    if prediction[0]==1:
        print("\nLoan Approved.")
    else:
        print("\nLoan Rejected.")

def main():
    model=load_model()
    new_data=get_user_input()
    predict(model,new_data)

if __name__=="__main__":
    main()