import pandas as pd
import joblib

MODEL_PATH="models/titanic_model.pkl"

def load_model():
    data=joblib.load(MODEL_PATH)
    model=data["model"]
    encoder=data["encoder"]
    return model,encoder

def get_user_input(encoder):
    try:
        pclass=int(input("Enter Passenger Class (1/2/3) :"))
        sex=input("Enter Sex(male/female) :")
        age=float(input("Enter Age :"))
        sibsp=int(input("Enter Siblings/Spouses :" ))
        parch=int(input("Enter Parents/children :"))
        fare=float(input("Enter Fare :"))
        sex_encoder=encoder.transform([sex])[0]
        new_data=pd.DataFrame([[pclass,sex_encoder,age,sibsp,parch,fare]]
                              ,columns=["Pclass","Sex","Age","SibSp","Parch","Fare"])
        return new_data
    except ValueError:
        print("Please enter valide input")
        return None

def predict_model(model,new_data):
    prediction=model.predict(new_data)
    if prediction[0]==1:
        print("Passenger will Survive.")
    else:
        print("Passenger will Not Survive.")

def main():
    try:
        model,encoder=load_model()
        new_data=get_user_input(encoder)
        if new_data is not None:
            predict_model(model,new_data)
    except Exception as e:
        print("Prediction Error :",e)

if __name__=="__main__":
    main()
