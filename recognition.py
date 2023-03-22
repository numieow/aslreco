import tensorflow as tf
from tensorflow.python import keras

import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import scipy as sp
import pandas as pd

#Importing data from data/sign_mnist_train.csv
df1 = pd.read_csv('data\sign_mnist_train.csv', delimiter=',')
print(df1)