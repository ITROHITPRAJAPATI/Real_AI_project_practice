import joblib

def load_model():
    model=joblib.load("models/spam_model.pkl")
    vectorizer=joblib.load("Models/vectorizer.pkl")
    return model,vectorizer

def get_message():
    return input("Enter Message :")

def predict_message(model,vectorizer,message):
    text=vectorizer.transform([message])
    prediction=model.predict(text)
    print("\n----- Spame Prediction our enter message-----")
    if prediction[0]==1:
        print("\nSpam Message.")
    else:
        print("\nNot Spam.")

def main():
    model,vectorizer=load_model()
    message=get_message()
    predict_message(model,vectorizer,message)

if __name__=="__main__":
    main()