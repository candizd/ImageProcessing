# -*- coding: utf-8 -*-

## Made using Fruits 360 dataset: 


import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import pickle

def preProcess(img):
    img = img /255   
    return img

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse_output = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

#Training Set
path = "Training"
train_images = []
train_labels = []

for class_dir in os.listdir(path):
    class_path = os.path.join(path, class_dir)
    # Check if the class_path is a directory
    classList = os.listdir(class_path)
    label = class_dir
    for i,image in enumerate(classList):
        check = int(len(os.listdir(class_path)) * 70 / 100)
        if i < check:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (32,32))
            train_images.append(img)
            train_labels.append(label)

path = "Test"    
test_images = []
test_labels = []

for class_dir in os.listdir(path):
    class_path = os.path.join(path, class_dir)
    # Check if the class_path is a directory
    classList = os.listdir(class_path)
    label = class_dir
    for i,image in enumerate(classList):
        check = int(len(os.listdir(class_path)) * 70 / 100)
        if i < check:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (32,32))
            test_images.append(img)
            test_labels.append(label)
            
            
x_test, y_test = test_images, test_labels
x_train, x_validation, y_train, y_validation = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 42)

x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))          
            
x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)
x_validation = x_validation.reshape(-1,32,32,3)     

y_train = onehot_labels(y_train)
y_test = onehot_labels(y_test)
y_validation = onehot_labels(y_validation)
            
dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.1,
                             rotation_range = 10)

dataGen.fit(x_train)

model = Sequential()

model.add(Conv2D(input_shape = (32,32,3), filters = 8, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D( filters = 16, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units = 256, activation = "relu" ))
model.add(Dropout(0.25))
model.add(Dense(units= len(os.listdir(path)), activation = "softmax" ))

model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])

batch_size = 64
            
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 15,steps_per_epoch = x_train.shape[0]//batch_size, shuffle = 1)
            
pickle_out = open("fruits_model_trained_new.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

## Evaluate 
hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()


score = model.evaluate(x_test, y_test, verbose = 1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
            
            
            
            
            
            
            
            
            
            
            