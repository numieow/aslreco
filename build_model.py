from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#An array containing the alphabet 
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 


#Importing data from data/sign_mnist_train.csv
training = pd.read_csv('data/sign_mnist_train.csv', delimiter=',') #Training data

#Gathering the data from the database
inputs_train = training.iloc[:, 1:].to_numpy() #Training inputs
targets_train = training['label'].to_numpy() #Training targets


#Normalize inputs
inputs_train = inputs_train / 255.0 #Normalizing the inputs
mean = np.mean(inputs_train) #Calculating the mean
std = np.std(inputs_train) #Calculating the standard deviation
inputs_train = (inputs_train - mean) / std #Normalizing the inputs
inputs_train = inputs_train.reshape(-1,28,28,1) #Reshaping the inputs to fit the model input shape of 28x28x1, the -1 is the batch size

#Using the data to train a Convolutional Neural Network 
model = keras.Sequential([ #Creating the model
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)), #Convolutional layer with 64 filters, 3x3 kernel size, relu activation function and input shape of 28x28x1
    keras.layers.MaxPooling2D(2,2),   #output_shape = math.floor((input_shape - pool_size) / strides) + 1 with strides = 2 and pool_size = 2
    keras.layers.Conv2D(64, (3,3), activation='relu'), #Convolutional layer with 64 filters, 3x3 kernel size and relu activation function
    keras.layers.MaxPooling2D(2,2), #output_shape = math.floor((input_shape - pool_size) / strides) + 1 with strides = 2 and pool_size = 2
    keras.layers.Flatten(), #Flattening the input to a 1D array
    keras.layers.Dense(128, activation='relu'),  #Dense layer with 128 neurons and relu activation function    max between 0 and x  128 is a power of 2 and is a common choice for the number of neurons in a hidden layer
    keras.layers.Dense(25, activation='softmax') #Dense layer with 25 neurons and softmax activation function  sum of all outputs = 1 why 25 ? 25 is the number of classes in the dataset and softmax is used for multiclass classification 
])

model.compile(optimizer='adam', # Compiling the model with the adam optimizer which is an extension of the stochastic gradient descent optimizer
                loss='sparse_categorical_crossentropy', #Using sparse categorical crossentropy as loss function since the targets are integers
                metrics=['accuracy']) #Using accuracy as metric to evaluate the model    Calculate how often the predictions equal the labels

model.fit(inputs_train, targets_train, epochs=4) #Training the model with the training data and targets for 10 epochs

#Testing the model
'''
test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
print('Test accuracy:', test_acc)'''

#Saving the model
model.save('model.h5') #Saving the model as a .h5 file

