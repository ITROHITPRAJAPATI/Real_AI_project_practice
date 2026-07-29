import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
df=pd.read_csv("spam.csv")
#print(df.head())
X=df["Message"]
y=df["Label"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,
    random_state=42
)
vectorizer=CountVectorizer()
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)
#print(X_train)
model=MultinomialNB()
model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# print(y_pred)
# accuracy=accuracy_score(y_test,y_pred)
# print("Accuracy : ",accuracy)
# new_email=["Congratulation! You Won a free mobile."]
# new_email=vectorizer.transform(new_email)
# prediction=model.predict(new_email)
# print(prediction)
joblib.dump(model,"spam_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

# loaded Model
loaded_model=joblib.load("spam_model.pkl")
loaded_vectorizer=joblib.load("vectorizer.pkl")
email=input("Enter Email : ")
email_vector=loaded_vectorizer.transform([email])
prediction=loaded_model.predict(email_vector)
if prediction[0]=="spam":
    print("🚨 Spam Email")
else:
    print("✔ Not Spam")
