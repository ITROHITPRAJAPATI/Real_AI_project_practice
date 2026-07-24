"""Project is House Price Pridection"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
import joblib
df=pd.read_csv("house_price.csv")
#print(df.head())
x=df[["Area","Bedrooms","Age"]]
y=df["Price"]
# print(x)
# print(y)
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,
    random_state=42
)
model=LinearRegression()
model.fit(x_train,y_train)
joblib.dump(model,"house_price_model.pkl")
y_pred=model.predict(x_test)
# print(y_pred)
# print(y_test.values)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("R2 Score :",r2)
print("Mean Absolute Error : ",mae)
loaded_model=joblib.load("house_price_model.pkl")

#input user 
def get_input():
    try:
        area=float(input("Enter House Area : "))
        bedrooms=int(input("Enter Bedrooms : "))
        age=int(input("Enter House Age : "))

        if area<=0 or bedrooms<=0 or age<0:
            print("invalid Input")
            return None
        return area,bedrooms,age
    except ValueError:
        print("Please enter valide numeric values.")
        return None
def predict_price(model,area,bedrooms,age):   
        new_data=pd.DataFrame({"Area":[area],"Bedrooms":[bedrooms],"Age":[age]})
        prince=model.predict(new_data)
        print("\n------House Price Prediction -------")
        print("\nPrediction House Price : ",round(prince[0],2))

def main():
    data=get_input()
    if data is None:
         return 
    area,bedrooms,age=data
    predict_price(loaded_model,area,bedrooms,age)
    print("House Price Prediction Completed Successfully")

main()