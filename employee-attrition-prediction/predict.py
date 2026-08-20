import joblib
import pandas as pd
MODEL_PATH="models/employee_attrition_model.pkl"

def load_model():
    model=joblib.load(MODEL_PATH)
    return model

def predict_attrition(model):
    age=int(input("Enter age :"))
    job_satisfaction=int(input("Enter job Satisfaction (1-4) :"))
    monthly_income=int(input("Enter Monthly Incime : "))
    year_at_company=int(input("Enter Years At Company :"))
    overtime=int(input("Ehter OverTime (1=Yes,0=No) :"))
    job_level=int(input("Enter Job Level (1-5) :"))

    data=pd.DataFrame({"Age":[age],"JobSatisfaction":[job_satisfaction],"MonthlyIncome":[monthly_income],"YearsAtCompany":[year_at_company],
                     "OverTime":[overtime],"JobLevel":[job_level]})
    
    prediction=model.predict(data)
    if prediction==1:
        print("Prediction: Employee may leave the Company.")
    else:
        print("Prediction: Employee may stay in the Company.")

def main():
    model=load_model()
    predict_attrition(model)

if __name__=="__main__":
    main()