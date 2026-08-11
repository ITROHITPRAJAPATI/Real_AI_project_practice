import joblib
from config import MODEL_PATH

def save_model(model,scaler):
    joblib.dump((model,scaler),MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)
