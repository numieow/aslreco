import tensorflow as tf
from tensorflow.python import keras

import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import scipy as sp
import pandas as pd
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

#An array containing the alphabet 
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


#Importing data from data/sign_mnist_train.csv
training = pd.read_csv('data\sign_mnist_train.csv', delimiter=',')
testing = pd.read_csv('data\sign_mnist_test.csv', delimiter=',')

#Gathering the data from the database
inputs_train = training.iloc[:, 1:].to_numpy()
targets_train = training['label'].to_numpy()

inputs_test = testing.iloc[:, 1:].to_numpy()
targets_test = testing['label'].to_numpy()


#Normalize inputs
inputs_train = inputs_train / 255.0
inputs_test = inputs_test / 255.0
inputs_train = inputs_train.reshape(-1,28,28,1)
inputs_test = inputs_test.reshape(-1,28,28,1)

#Using the data to train a Convolutional Neural Network 
model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(25, activation='softmax')
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(inputs_train, targets_train, epochs=10)

#Testing the model
test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
print('Test accuracy:', test_acc)

#Saving the model
model.save('model.h5')

#Predicting the test data
predictions = model.predict(inputs_test)

#Plotting the results
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(inputs_test[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(alphabet[np.argmax(predictions[i])])
plt.show()

#Saving the model
model.save('model.h5')

