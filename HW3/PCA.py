import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#import datasets
df_1 = pd.read_csv('dataI.csv')
df_2 = pd.read_csv('dataII.csv')
df_3 = pd.read_csv('dataIII.csv')
df_4 = pd.read_csv('dataIV.csv')
df_5 = pd.read_csv('dataV.csv')
iris = pd.read_csv('iris.csv')

#Get the mean of each column for data reconstruction
avg_1 = df_1.mean().values
avg_2 = df_2.mean().values
avg_3 = df_3.mean().values
avg_4 = df_4.mean().values
avg_5 = df_5.mean().values


#Normalize the datasets so that they have zero means
data_1 = (df_1 - df_1.mean()).values
data_2 = (df_2 - df_2.mean()).values
data_3 = (df_3 - df_3.mean()).values
data_4 = (df_4 - df_4.mean()).values
data_5 = (df_5 - df_5.mean()).values
iris = (iris - iris.mean()).values


from scipy.spatial.distance import sqeuclidean
#This function calculates
def comp_err(A, B):
    error = 0.0
    for elemA, elemB in zip(A,B):
        error += sqeuclidean(elemA, elemB)
    return error / A.shape[0]

data_list =[data_1, data_2, data_3, data_4, data_5]
err_matrix = np.zeros([5,5])


for data in range(5):
    for i in range(5):
        pca = PCA(n_components = i, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
        pc = pca.fit_transform(data_list[data])
        reform = pca.inverse_transform(pc) + avg_1
        err_matrix[data, i] = comp_err(reform, iris)
