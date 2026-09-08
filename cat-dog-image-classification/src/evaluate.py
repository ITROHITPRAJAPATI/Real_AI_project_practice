import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report

from data_loader import load_image,split_data

MODEL_PATH="models/cats_dogs_model.keras"

#load training model
model=tf.keras.models.load_model(MODEL_PATH)

# load data set 
images,labels=load_image()
X_train,X_test,y_train,y_test=split_data(images,labels)

# Evaluated model
loss,accuracy=model.evaluate(X_test,y_test,verbose=0)
print(f"Test Accuracy :{accuracy*100:.2f}")
print(f"Test Loss :{loss:.4f}")

# prediction
prediction=model.predict(X_test,verbose=0)
y_pred=(prediction>0.5).astype(int).flatten()

# confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:")
print(cm)

#classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,y_pred,target_names=["Cat","Dog"]
    )
)

