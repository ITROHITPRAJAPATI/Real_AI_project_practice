import pandas as pd
from utils import load_model
from logger import logger

def get_user_input():
    try:
        age=int(input("Enter Age :"))
        charges=float(input("Enter Monthly Charges :"))
        tenure=int(input("Enter Tenure :"))
        return pd.DataFrame({"Age":[age],"MonthlyCharges":[charges],"Tenure":[tenure]})
    except ValueError:
        print("Please enter valide numerical values.")
        return None

def predict_customer(model,new_data):
    prediction=model.predict(new_data)
    if prediction[0]==1:
        print("\nCustomer Will Churn.")
    else:
        print("\nCustomer Will Not Churn.")

def main():
    try:
        model=load_model()
        new_data=get_user_input()
        if new_data is not None:
            predict_customer(model,new_data)
    except Exception as e:
        logger.error(f"Prediction Error :{e}")
        print("Error :",e)

if __name__=="__main__":
    main()