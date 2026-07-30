from sklearn.datasets import load_iris
iris=load_iris(as_frame=True)
df=iris.frame
df.to_csv("iris.csv",index=False)
print("iris.csv Successfully created")