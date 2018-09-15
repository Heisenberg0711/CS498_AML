import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit


#This function is used to convert string to values
def label_income(row):
    if row[14] == " <=50K":
        return -1
    if row[14] == " >50K":
        return 1

#Detect NA values and mark them as NA
df = pd.read_csv('train.data', header = None)
df = df.dropna(axis=0,how='any')
labels = df.apply(lambda row: label_income (row),axis=1).values

#Exclude non-continuous values
df = df.drop(columns = [1,3,5,6,7,8,9,13])
features = df.drop(columns=14).values
features = features.astype(float)

#Normalizing features
for col in range(features.shape[1]):
    a = features[:,col]
    features[:,col] = (a - a.mean()) / a.std()

#Get the test and train data sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1)

#An array of regulation variables
lamda_vals = np.array([1e-3,1e-2,1e-1,1])

#Define constants for calculating stepsize = a / (epoch + b)
m = 6
n = 2

#Number of steps per season
N_steps = 300

#This function calculates gradient and updates the values of a and b
#returns a tuple of a vector and b value
def gradient(a, b, X, Y,lam):
    mat = Y * (np.dot(X,a) + b)
    idx1 = np.where(mat >= 1)
    idx0 = np.where(mat < 1)

    posA = len(idx1[1])*lam*a
    negA = len(idx0[0])*lam*a - np.sum(Y[idx0]*X[idx0], axis = 0)
    negB = -np.sum(Y[idx0])
    return (posA+negA, negB)


#This function will give the results of the model
def get_accu(a, b, val_features, val_labels):
    prediction = val_features.dot(a)
    result = np.where(prediction > 0, 1, -1).reshape(50,)
    print(result.shape)
    print(val_labels.shape)
    print(np.sum(result == val_labels))
    accuracy = np.sum(result==val_labels) / val_labels.shape[0]
    return accuracy



lam = 1e-2

for epoch in arange(1,51):

    ida = m / (epoch + n)
    a = np.random.random([6, 1])
    b = np.random.random()

    for step in range(300):
        #Create a mask to choose batches for each epoch
        msk = np.array(range(0,len(train_labels)))
        np.random.shuffle(msk)

        held_features = np.array(train_features[msk[-50:]])
        held_labels = np.array(train_labels[msk[-50:]])
        Index_array = np.array_split(msk[:-50], N_steps)

        #for step in range(1):



        batch_size = len(Index_array[0])
        X = train_features[Index_array[0]]
        Y = train_labels[Index_array[0]].reshape(batch_size, 1)
        gradA, gradB = gradient(a,b,X,Y,0.1)
        a = a - ida * (1/batch_size)*gradA
        b = b - ida * (1/batch_size)*gradB
