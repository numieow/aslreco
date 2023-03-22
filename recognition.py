import tensorflow as tf
from tensorflow.python import keras

import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import scipy as sp
import pandas as pd

#Importing data from data/sign_mnist_train.csv
training = pd.read_csv('data\sign_mnist_train.csv', delimiter=',')
testing = pd.read_csv('data\sign_mnist_test.csv', delimiter=',')

inputs_train = training.iloc[:, 1:].to_numpy()
targets_train = training['label'].to_numpy()

inputs_test = testing.iloc[:, 1:].to_numpy()
targets_test = testing['label'].to_numpy()

#The different labels
print(inputs_train['label'].unique())

#Normalize inputs
inputs_train = inputs_train / 255.0
inputs_test = inputs_test / 255.0



