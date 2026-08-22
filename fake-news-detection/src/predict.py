import joblib
MODEL_PATH="models/fake_news_model.pkl"
VECTORIZER_PATH="models/tfidf_vectorizer.pkl"

def load_model():
    model=joblib.load(MODEL_PATH)
    vectorizer=joblib.load(VECTORIZER_PATH)
    return model,vectorizer

def predict_news(model,vectorizer):
    news=input("Enter News :")
    news_vectorized=vectorizer.transform([news])
    prediction=model.predict(news_vectorized)
    print("Prediction :",prediction[0])

def main():
    model,vectorizer=load_model()
    predict_news(model,vectorizer)

if __name__=="__main__":
    main()