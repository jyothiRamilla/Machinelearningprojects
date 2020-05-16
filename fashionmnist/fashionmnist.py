# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:09:48 2020

@author: Lenovo
"""

from sklearn.model_selection  import train_test_split
import numpy as np
import pandas as pd
from keras.utils import to_categorical


fashion_train = pd.read_csv("data/fashion-mnist_train.csv")
fashion_test = pd.read_csv("data/fashion-mnist_test.csv")

fashion_train.head()

fashion_test.head()

X= np.array(fashion_train.iloc[:,1:])

#y= np.array(fashion_test.iloc[:,0])
y = to_categorical(np.array(fashion_train.iloc[:, 0]))


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=0)


X_test = np.array(fashion_test.iloc[:,1:])
y_test = to_categorical(np.array(fashion_test.iloc[:,0]))
img_rows =28
img_cols =28

X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test  = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
X_val   = X_val.reshape(X_val.shape[0],img_rows,img_cols,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_val=X_val.astype('float32')


X_train /= 255
X_test /=255
X_val /=255

#import matplotlib.pyplot as plt

#plt.hist(fashion_train,bins=50)

#plt.bar(fashion_train,2,color="red")



from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D,Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import keras

model = Sequential()



from keras.layers.normalization import BatchNormalization

batch_size = 256
num_classes = 10
epochs = 50

#input image dimensions
img_rows, img_cols = 28, 28

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'))
"""
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
"""
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

"""model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
"""


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

"""history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))"""

#score = model.evaluate(X_test, y_test, verbose=0)

history= model.fit(X_train, y_train, batch_size=256, epochs=10, validation_data=(X_val, y_val))


score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


import matplotlib.pyplot as plt

history_dict = history.history
print(history_dict.keys())
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
plt.show()


#get the predictions for the test data
predicted_classes = model.predict_classes(X_test)

np.unique(predicted_classes)

#get the indices to be plotted
y_true = fashion_test.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]

num_classes =10
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.tight_layout()
    
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
    plt.tight_layout()
    
test_im = X_train[154]
plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')
plt.show()

from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(input=model.input, output=layer_outputs)
activations = activation_model.predict(test_im.reshape(1,28,28,1))

first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

layer_names = []
for layer in model.layers[:-1]:
    layer_names.append(layer.name) 
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    if layer_name.startswith('conv'):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        








