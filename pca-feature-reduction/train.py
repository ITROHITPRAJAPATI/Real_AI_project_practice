import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris=load_iris()
df=pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
#print(df.head())
pca=PCA(n_components=2)
new_data=pca.fit_transform(df)
#print(new_data)
print("Original Shape :",df.shape)
print("Reduced Shape :",new_data.shape)
#Explained Variance Ratio
print("Expalain Variance Ratio :")
print(pca.explained_variance_ratio_)
#Total Information
print("\n Total Information Preserved ")
print(pca.explained_variance_ratio_.sum())