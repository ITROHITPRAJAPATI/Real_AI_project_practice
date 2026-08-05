import pandas as pd
import joblib

def load_model():
    return joblib.load("models/diabetes_model.pkl")

def get_user_input():
    try:
        age=int(input("Enter Age :"))
        glucose=int(input("Enter Glucose :"))
        bmi=int(input("Enter BMI :"))
        return pd.DataFrame({"Age":[age],"Glucose":[glucose],"BMI":[bmi]})
    except ValueError:
        print("Pleas enter valid numeric Values.")
        exit()

def predict(model,new_data):
    prediction=model.predict(new_data)
    if prediction[0]==1:
        print("\nDiabetes Detected.")
    else:
        print("\nNo Diabetes")

def main():
    model=load_model()
    new_data=get_user_input()
    predict(model,new_data)

if __name__=="__main__":
    main()