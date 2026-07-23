# import pandas as pd
# df=pd.read_csv("salary.csv")
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

"""LinearRegration (Salary Prediction)"""
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# df=pd.read_csv("salary.csv")
# x=df[["Experience","Age"]]
# y=df["Salary"]
# x_train,x_test,y_train,y_test=train_test_split(
#     x,y,test_size=0.2,
#     random_state=42
# )
# model=LinearRegression()
# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# print("Prediction :",prediction)
# print("Actual Values :",y_test.values)
# from sklearn.metrics import r2_score,mean_absolute_error
# r2=r2_score(y_test,prediction)
# print("R2 Score : ",r2)
# mae=mean_absolute_error(y_test,prediction)
# print("MAE : ",mae)

"""salary Prediction Input values by user"""
# import pandas as pd
# import joblib 
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# df=pd.read_csv("salary.csv")
# x=df[["Experience","Age"]]
# y=df["Salary"]
# x_train,x_test,y_train,y_test=train_test_split(
#     x,y,test_size=0.2,
#     random_state=42
# )
# model=LinearRegression()
# model.fit(x_train,y_train)
# joblib.dump(model,"salary_model.pkl")
# print("Model Training succesfully ")

#User input and Prediction
# import pandas as pd
# import joblib

# model=joblib.load("salary_model.pkl")
# experience=float(input("Enter Experirnce (years) : "))
# age=float(input("Enter Age : "))
# if (experience<0 or experience>50 or age<18 or age>70):
#     print("Invalid Input")
#     exit()
# new_data=pd.DataFrame({"Experience":[experience],"Age":[age]})
# salary=model.predict(new_data)
# print("\n--------- Salary Prediction Result ---------")
# print("Prediction Salary : rupees ",round(salary[0],2))
# print("Thanks You For Using Salary Prediction System")

"""Professional tarika """
import pandas as pd
import joblib 
model=joblib.load("salary_model.pkl")
#user input funtion
def get_input():
    try:
        experience=float(input("Enter Experience (year) : "))
        age=float(input("Enter Age : "))
        if experience<0 or experience>50 or age<18 or age>70:
            print("Invalid Input")
            return None
        return experience,age
    except ValueError:
        print("Pleass enter Only numbet ")
        return None

def predict_salary(model,experience,age):
    new_data=pd.DataFrame({"Experience":[experience],"Age":[age]})
    salary=model.predict(new_data)
    print("\n-------Salary Prediction Result -------")
    print("Predicted Salary in Rupees :",round(salary[0],2))

def main():
    data=get_input()
    if data is None:
        return
    experience,age=data
    predict_salary(model,experience,age)
    print("Thanks You for Using Salary Prediction System ")

main()
print("Project Update")