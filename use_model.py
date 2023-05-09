from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Use the model.h5 model to predict from data\sign_mnist_test.csv
#An array containing the alphabet
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#Importing data from data/sign_mnist_test.csv
testing = pd.read_csv('data/sign_mnist_test.csv', delimiter=',')
inputs_test = testing.iloc[:, 1:].to_numpy()
targets_test = testing['label'].to_numpy()
#Importing data from data/sign_mnist_train.csv
training = pd.read_csv('data/sign_mnist_train.csv', delimiter=',')
inputs_train = training.iloc[:, 1:].to_numpy()
targets_train = training['label'].to_numpy()


#Normalize inputs
inputs_train = inputs_train / 255.0
inputs_test = inputs_test / 255.0
mean = np.mean(inputs_train)
std = np.std(inputs_train)
inputs_test = (inputs_test - mean) / std
inputs_test = inputs_test.reshape(-1,28,28,1)

#Load the model
model = keras.models.load_model('model.h5')

#Testing the model
test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
print('Test accuracy:', test_acc)

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

#Save the plot
plt.show()
plt.savefig('results.png')
