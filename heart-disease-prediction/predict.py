import pandas as pd
import joblib

def load_model():
    return joblib.load("models/heart_model.pkl")

def get_input():
    age=int(input("Enter Age :"))
    bp=int(input("Enter BP :"))
    Cholesterol=int(input("Enter Cholesterol :"))
    return pd.DataFrame({"Age":[age],"BP":[bp],"Cholesterol":[Cholesterol]})

def predict(model,new_data):
    prediction=model.predict(new_data)
    if prediction[0]==1:
        print("\nHeart Disease Risk.")
    else:
        print("\nNo Heart Disease.")

def main():
    model=load_model()
    new_data=get_input()
    predict(model,new_data)

if __name__=="__main__":
    main()