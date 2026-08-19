import joblib

MODEL_PATH="model/sentiment_model.pkl"
VECTORIZER_PATH="model/tfidf_vectorizer.pkl"

def load_model():
    model=joblib.load(MODEL_PATH)
    vectorizer=joblib.load(VECTORIZER_PATH)
    return model,vectorizer

def predict_sentiment(model,vectorizer):
    review=input("Enter your review :")
    review_vectorized=vectorizer.transform([review])
    prediction=model.predict(review_vectorized)
    print("Sentiment :",prediction[0])

def main():
    model,vectorizer=load_model()
    predict_sentiment(model,vectorizer)

if __name__=="__main__":
    main()