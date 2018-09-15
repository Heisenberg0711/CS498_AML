import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit


#This function is used to convert string to values
def label_income(row):
    if row[14] == " <=50K":
        return 0
    if row[14] == " >50K":
        return 1

#Detect NA values and mark them as NA
df = pd.read_csv('train.data', header = None, na_values=" ?")
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
a = 6
b = 2

#Number of steps per season
N_steps = 300

#This function calculates gradient and updates the values of a and b
#returns an numpy array of a and b
def gradient(a, b, X, Y,lam):
    if Y*(a.T*X + b) < 1:
        return np.array([lam*a - Y*X, -Y])
    else
        return np.array([lam*a, 0])


msk = np.array(range(0,len(train_labels)))
np.random.shuffle(msk)

batch = train_features[msk[0:10],:]
