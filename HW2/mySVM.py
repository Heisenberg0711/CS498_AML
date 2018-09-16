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


def parse(row):
    return "'" + row + "'"

#Detect NA values and mark them as NA
df = pd.read_csv('train.data', header = None)
df = df.dropna(axis=0,how='any')
labels = df.apply(lambda row: label_income(row),axis=1).values

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

#Define constants for calculating stepsize = a / (epoch + b)
m = 1
n = 50

#Number of steps per season and regularization constants
N_steps = 300
lambda_vals = np.array([1e-3,0.002, 0.003, 1e-2,1e-1,1])

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
def get_accu(a, b, val_features, val_labels, size):
    prediction = val_features.dot(a) + b
    result = np.where(prediction > 0, 1, -1).reshape(size,)
    accuracy = np.sum(result==val_labels) / val_labels.shape[0]
    return accuracy


def SVM(train_features, train_labels, lam):
    a = np.random.random([6, 1])
    b = np.random.random()

    #Empty accuracy array and magnitude array
    accu_array = []
    mag_array = []

    for epoch in range(1,51):
        #Calculate stepsize based on epoch
        ida = m / (0.01 * epoch + n)

        #Create a mask to choose batches for each epoch
        msk = np.array(range(0,len(train_labels)))
        np.random.shuffle(msk)

        held_features = np.array(train_features[msk[-50:]])
        held_labels = np.array(train_labels[msk[-50:]])
        Index_array = np.array_split(msk[:-50], N_steps)

        for step in range(300):
            batch_size = len(Index_array[step])
            X = train_features[Index_array[step]]
            Y = train_labels[Index_array[step]].reshape(batch_size, 1)
            gradA, gradB = gradient(a, b, X, Y, lam)
            a = a - ida * (1/batch_size)*gradA
            b = b - ida * (1/batch_size)*gradB

            if (step % 30 == 0):
                accu_array.append(get_accu(a, b, held_features, held_labels, 50))
                mag_array.append(a.T.dot(a)[0,0])
    return (a, b, accu_array, mag_array)


#Call the SVM function and plot accuracy and magnitudes
accu_all = []
mag_all = []
steps = np.arange(500)

for reg in lambda_vals:
    a, b, accu, mag = SVM(train_features, train_labels, reg)
    accu_all.append(accu)
    mag_all.append(mag)


#Making accuracy and magnitude plots
plt.figure(1)
line1, = plt.plot(steps, accu_all[0], label='1e-3')
line2, = plt.plot(steps, accu_all[1], label='1e-2')
line3, = plt.plot(steps, accu_all[2], label='1e-1')
line4, = plt.plot(steps, accu_all[3],label='1')
plt.legend(handles=[line1, line2, line3, line4])
plt.title("Accuracy vs. Steps")

plt.figure(2)
line5, = plt.plot(steps, mag_all[0], label='1e-3')
line6, = plt.plot(steps, mag_all[1], label='1e-2')
line7, = plt.plot(steps, mag_all[2], label='1e-1')
line8, = plt.plot(steps, mag_all[3],label='1')
plt.legend(handles=[line1, line2, line3, line4])
plt.title("Magnitude vs. Steps")



#Use the given test data to validate the bodel
df_val = pd.read_csv('test.data', header = None)
df_val = df_val.dropna(axis=0, how='any')
val_features = df_val.drop(columns = [1,3,5,6,7,8,9,13]).values
val_features = val_features.astype(float)

#Normalizing validation features
for col in range(val_features.shape[1]):
    item = val_features[:,col]
    val_features[:,col] = (item - item.mean()) / item.std()

val_pred = val_features.dot(a) + b
val_result = np.where(val_pred > 0, 1, -1).reshape(len(val_pred),)
val_result = list(val_result)

for i in range(len(val_result)):
    if val_result[i] == 1:
        val_result[i] = ">50K"
    else:
        val_result[i] = "<=50K"

id = np.arange(0,len(val_result))
dict = {'Example': id, 'Label': val_result}
df = pd.DataFrame(data = dict)
df['Example'] = df['Example'].apply(str)
df['Example'] = df['Example'].apply(parse)
df.to_csv("hqiu9.csv", index = False)
