import pandas as pd
import joblib 
#Model Load
model=joblib.load("models/salary_model.pkl")
experience=float(input("Enter Experience :"))
new_employee=pd.DataFrame({"Experience":[experience]})
salary=model.predict(new_employee)
print("\nPrediction Salary :",salary[0])