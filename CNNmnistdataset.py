# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:50:08 2020

@author: Lenovo
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D,Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score



##Load the dataset

(X_train,y_train),(X_test,y_test)= mnist.load_data()

print("Shape of X_train",X_train.shape)
print("Shape of y_train",y_train.shape)
print("Shape of X_test",X_test.shape)
print("Shape of y_test",y_test.shape)

##Neural network for MNIST dataset

"""Steps:
   1. Flatten the input image dimensions to 1D (width pixels x height pixels)
   2.Normalize the image pixel values (divide by 255)
   3.One-Hot Encode the categorical column
   4.Build a model architecture (Sequential) with Dense layers
   5.Train the model and make predictions
"""
#Flattenning the data into 1D 784 px

X_train = X_train.reshape(60000,784)
X_test =  X_test.reshape(10000,784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Normalize the image pixel values
X_train/=255
X_test/=255

#One-Hot Encoding  using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()

# hidden layer
model.add(Dense(100, input_shape=(784,), activation='relu'))

# output layer
model.add(Dense(10, activation='softmax'))

# looking at the model summary
model.summary()

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.compile()
# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))


####CONVOLUTIONAL NEURAL NETWORK
"""One major advantage of using CNNs over NNs is that you do not need to flatten the input images to 1D as they are capable of working with image data in 2D.
 This helps in retaining the “spatial” properties of images"""

##Loading
(A_train,b_train),(A_test,b_test)= mnist.load_data()

##Building input vector to form 28*28 pixels
A_train = A_train.reshape(A_train.shape[0],28,28,1)
A_test  = A_test.reshape(A_test.shape[0],28,28,1)
A_train = A_train.astype('float32')
A_test  = A_test.astype('float32')

##Normalizing the data to help the training
A_train/= 255
A_test/= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", b_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", b_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(A_train, b_train, batch_size=128, epochs=10, validation_data=(A_test, b_test))


