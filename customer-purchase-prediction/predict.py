import pandas as pd 
from utils import load_model
from logger import logger

def get_user_input():
    try:
        age=int(input("Enter Age :"))
        salary=int(input("Enter Salary :"))
        return pd.DataFrame({"Age":[age],"Salary":[salary]})
    except ValueError:
        print("Pleas enter valide number .")
        return None

def predict_model(model,scaler,new_data):
    new_data_sacled=scaler.transform(new_data)
    prediction=model.predict(new_data_sacled)
    if prediction[0]==1:
        print("\nCustomer will Purchase.")
    else:
        print("\nCustomer will not purchage.")

def main():
    try:
        model,scaler=load_model()
        new_data=get_user_input()
        predict_model(model,scaler,new_data)
    except Exception as e:
        logger.error(f"Prediction Error {e}")
        print("Error",e)

if __name__=="__main__":
    main()