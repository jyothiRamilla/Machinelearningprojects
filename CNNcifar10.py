# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:08:47 2020

@author: Lenovo
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils




(X_train,y_train),(X_test,y_test)=cifar10.load_data()

X_train = X_train.reshape(X_train.shape[0],32,32,3)
X_test  = X_test.reshape(X_test.shape[0],32,32,3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255


n_classes=10

print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

model = Sequential()

model.add(Conv2D(70 , kernel_size= (3,3) ,strides= (1,1) ,padding="same" ,activation="relu", input_shape=(32,32,3)))


model.add(Conv2D(80, kernel_size= (3,3),strides=(1,1),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(100, kernel_size=(3,3),strides=(1,1),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))




