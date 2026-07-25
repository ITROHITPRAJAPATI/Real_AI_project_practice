""" Project: Car Price Prediction """
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
df=pd.read_csv("car_price.csv")
# print(df.head())
# print(df.shape)
# print(df.columns)
x=df[["Year","KM_Driven"]]
y=df["Price"]
# print("\nFeature (x): ")
# print(x)
# print("\nTarget (y):")
# print(y)
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,
    random_state=42
)
# print("x_train shape",x_train.shape)
# print("x_test shape",x_test.shape)
# print("y_train shape",y_train.shape)
# print("y_test shape",y_test.shape)
model=LinearRegression()
model.fit(x_train,y_train)
joblib.dump(model,"car_price_model.pkl")
#y_pred=model.predict(x_test)
#print(y_pred)
#r2=r2_score(y_test,y_pred)
#print("R2 Score : ",r2)
#mae=mean_absolute_error(y_test,y_pred)
#print("Mean Absolute Error : ",mae)
load_model=joblib.load("car_price_model.pkl")
def get_input():
    try:
        year=int(input("Enter year : "))
        km=int(input("Enter KM Driven : "))
        if year<=0 or km<0:
            print("Invalid Input")
            return None
        return year,km
    except ValueError:
        print("Pleas enter valide numeric value.")
        return None
def predict_price(model,year,km):
    new_data=pd.DataFrame({"Year":[year],"KM_Driven":[km]})
    price=model.predict(new_data)
    print("\n------Car Price Prediction-----")
    print("Prediction Car Price :",round(price[0],2))
def main():
    data=get_input()
    if data is None:
        return 
    year,km=data
    predict_price(load_model,year,km)
    print("Car Price Prediction Completed Successfully")

main()