import joblib 
import pandas as pd
#Load traind model
model=joblib.load("models/house_price_model.pkl")
try:
    area=float(input("Enter Area :"))
    bedrooms=int(input("Enter Bedrooms :"))
    age=int(input("Enter Age :"))
    if age<0:
        print("House age can not be negative.")
        exit()
    #Create input Data Frame
    new_house=pd.DataFrame({"Area":[area],"Bedrooms":[bedrooms],"Age":[age]})
    #Predict House Price
    prediction=model.predict(new_house)
    print("\n Prediction House Price :",prediction[0])
except ValueError:
    print("Pleas Enter numberic value")
except FileNotFoundError:
    print("Model file not found.")