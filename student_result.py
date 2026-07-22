"""Student Result Prediction """
import pandas as pd
import joblib
# from sklearn.tree import DecisionTreeClassifier
# data={
#     "Hours":[2,3,4,5,6,7,8,9],
#     "Attendance":[60,65,70,75,80,85,90,95],
#     "Assignment":[50,55,60,65,70,75,80,85],
#     "Result":[0,0,0,1,1,1,1,1]
# }
# df=pd.DataFrame(data)
# x=df[["Hours","Attendance","Assignment"]]
# y=df["Result"]
# model=DecisionTreeClassifier()
# model.fit(x,y)
# joblib.dump(model,"student_model.pkl")
# print("Model Save Succesfully")

import sys
print("======================\n" \
"Student Result Prediction\n" \
"=======================")
#Load model
model=joblib.load("student_model.pkl")
#Input by User 
hours=int(input("Enter Ours Hours : "))
attendance=int(input("Enter Ours Attendance : "))
assignment=int(input("Enter Our Assignment : "))

if hours<0 or hours>24 or attendance>100 or attendance<0 or assignment>100 or assignment<0:
    print("Invalid Input")
    sys.exit()
#prediction 
prediction=model.predict(pd.DataFrame({"Hours":[hours],"Attendance":[attendance],"Assignment":[assignment]}))
print("Prediction Succesfully")
#Result
if prediction[0]==1:
    print("Pass")
    print("Congratulation")
else:
    print("Fail")
    print("Work Hard")
print("Thank You For Using Our AI Projerct ")