import pandas as pd
from sklearn.cluster import KMeans
import joblib
#csv file load
df=pd.read_csv("data/customer.csv")
model=KMeans(n_clusters=2,random_state=42)
model.fit(df)
# print(model.labels_)
# #cluster label
# df["Cluster"]=model.labels_
# print(df)
# #cluster center
# print("\n cluster centers:")
# print(model.cluster_centers_)
#save model
joblib.dump(model,"models/kmeans_model.pkl")
print("\nModel Saved Successfully")
# new_customer=[[30,45000]]
# prediction=model.predict(new_customer)
# print("Prediction Cluster : ",prediction[0])