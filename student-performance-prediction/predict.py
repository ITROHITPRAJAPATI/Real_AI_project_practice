import pandas as pd
from utils import load_model
from sklearn.metrics import accuracy_score

model=load_model()
hours=float(input("Enter Study Hours :"))
attendance=input("Enter Attendance :")
new_data=pd.DataFrame({"Hours":[hours],"Attendance":[attendance]})
prediction=model.predict(new_data)
print("Prediction Marks :",prediction[0])