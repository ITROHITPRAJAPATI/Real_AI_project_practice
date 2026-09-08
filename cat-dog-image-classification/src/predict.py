import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH="models/cats_dogs_model.keras"
IMAGE_PATH="data/raw/cats/cat.4001.jpg"

model=tf.keras.models.load_model(MODEL_PATH)
image=cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError("Image not found")
image=cv2.resize(image,(128,128))
image=image/255.0
image=np.expand_dims(image,axis=0)
prediction=model.predict(image)
probability=prediction[0][0]
if probability>0.5:
    label="Dog 🐶"
    confidence=probability*100
else:
    label="Cat 😺"
    confidence=(1-probability)*100

print(f"prediction:{label}")
print(f"confidence:{confidence:.2f}%")