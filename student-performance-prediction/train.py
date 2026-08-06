import pandas as pd
from sklearn.linear_model import LinearRegression
from config import DATA_PATH
from utils import save_model
from logger import logger

df=pd.read_csv(DATA_PATH)
X=df[["Hours","Attendance"]]
y=df["Marks"]
model=LinearRegression()
model.fit(X,y)
save_model(model)
logger.info("Model Trained Successfully")
print("Model Saved Successfully.")