import pandas as pd

DATA_PATH="data/raw/news.csv"
df=pd.read_csv(DATA_PATH)
print("Dataset Shape :",df.shape)
print("\nColumns :")
print(df.columns.tolist())
print("\nMissing Value:")
print(df.isnull().sum())
print("\nLabel Distribution :")
print(df["label"].value_counts())
print("\nDataset Validation Completed Successfully.")