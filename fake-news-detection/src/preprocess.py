import pandas as pd

INPUT_PATH="data/raw/news.csv"
OUTPUT_PATH="data/processed/clean_news.csv"

df=pd.read_csv(INPUT_PATH)
#Remove missing values
df=df.dropna()
#Remove Duplicate Rows
df=df.drop_duplicates()
#Clean text
df["text"]=df["text"].str.lower().str.strip()
#Saved Processed Data
df.to_csv(OUTPUT_PATH,index=False)
print("Preprocessing Completed Successfully.")
print("Processing Data Shape :",df.shape)