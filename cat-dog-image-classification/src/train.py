import os
import tensorflow as tf
from data_loader import load_image,split_data
from model import build_model

MODEL_PATH="models/cats_dogs_model.keras"
BEST_MODEL_PATH="models/model_best_cats_dogs_model.keras"

def main():
    # Load dataset
    images,labels=load_image()
    # Split dataset
    X_train,X_test,y_train,y_test=split_data(
        images,labels
    )
    model=build_model()
    # Early Stoping
    early_stopping=tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    # Save best model
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
    # train model
    history=model.fit(
        X_train,y_train,
        validation_data=(X_test,y_test),
        epochs=30,
        batch_size=32,
        callbacks=[
            early_stopping,checkpoint
        ]
    )
    # Save final Model
    os.makedirs("models",exist_ok=True)
    model.save(MODEL_PATH)
    print("Trainig completed successfully.")
    print(f"Final model save at:{MODEL_PATH}")
    print(f"best model saved at :{BEST_MODEL_PATH}")

if __name__=="__main__":
    main()