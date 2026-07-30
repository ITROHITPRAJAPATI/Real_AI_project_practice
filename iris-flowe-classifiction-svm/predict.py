import joblib
import pandas as pd
model=joblib.load("models/svm_model.pkl")
# new_data=pd.DataFrame([[5.1,3.5,1.4,0.2]],
#                       columns=["sepal length (cm)","sepal width (cm)",
#                                "petal length (cm)","petal width (cm)"])
# prediction=model.predict(new_data)
flower_name=["Setosa","Versicolor","Virginica"]
#print("Flower Name :",flower_name[prediction[0]])
try:
    sepal_length=float(input("Enter Sepal Length :"))
    sepal_width=float(input("Enter Sepal Width :"))
    petal_lenght=float(input("Enter Petal Length :"))
    petal_width=float(input("Enter Petal Width :"))
    new_data=pd.DataFrame([[sepal_length,sepal_width,petal_lenght,petal_width]],columns=["sepal length (cm)",
                                                                                         "sepal width (cm)",
                                                                                         "petal length (cm)",
                                                                                         "petal width (cm)"])
    prediction=model.predict(new_data)
    print("Predicted Flower :",flower_name[prediction[0]])
except ValueError:
    print("Please enter numeric values.")