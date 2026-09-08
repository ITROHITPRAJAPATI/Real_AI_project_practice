import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

CAT_DIR="data/raw/cats"
DOG_DIR="data/raw/dogs"
IMAGE_SIZE=128

def load_image():
    images=[]
    labels=[]
    # Load cat image
    for file_name in os.listdir(CAT_DIR):
        file_path=os.path.join(CAT_DIR,file_name)
        image=cv2.imread(file_path)
        if image is None:
            continue
        image=cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
        image=image/255.0
        images.append(image)
        labels.append(0)

    # Load Dog image
    for file_name in os.listdir(DOG_DIR):
        file_path=os.path.join(DOG_DIR,file_name)
        image=cv2.imread(file_path)
        if image is None:
            continue
        image=cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
        image=image/255.0
        images.append(image)
        labels.append(1)
    print("Cat images :",labels.count(0))
    print("Dog images :",labels.count(1))
    return np.array(images),np.array(labels)

def split_data(images,labels):
    X_train,X_test,y_train,y_test=train_test_split(
        images,labels,test_size=0.2,
        random_state=42,
    )        
    return X_train,X_test,y_train,y_test

if __name__=="__main__":
    images,labels=load_image()
    X_train,X_test,y_train,y_test=split_data(images,labels)
    print("Total images :",len(images))
    print("Train image :",len(X_train))
    print("Testing Image :",len(X_test))
    print("Data Loading Completed Successfully.")