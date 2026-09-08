import tensorflow as tf
from tensorflow.keras import layers,models

IMAGE_SIZE=128
def build_model():
    data_augmentation=tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])
    model=models.Sequential([
        layers.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
        data_augmentation,
        layers.Conv2D(32,(3,3),activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64,(3,3),activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(128,(3,3),activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1,activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

if __name__=="__main__":
    model=build_model()
    model.summary()