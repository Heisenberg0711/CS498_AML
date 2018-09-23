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
avg_iris = iris.mean().values

#Normalize the datasets so that they have zero means
data_1 = (df_1 - df_1.mean()).values
data_2 = (df_2 - df_2.mean()).values
data_3 = (df_3 - df_3.mean()).values
data_4 = (df_4 - df_4.mean()).values
data_5 = (df_5 - df_5.mean()).values
iris = iris.values


from scipy.spatial.distance import sqeuclidean
from sklearn.metrics import mean_squared_error
#This function calculates
def comp_err(A, B):
    error = 0.0
    for elemA, elemB in zip(A,B):
        error += sqeuclidean(elemA, elemB)
    return error / A.shape[0]

data_list =[data_1, data_2, data_3, data_4, data_5]
avg_list = [avg_1, avg_2, avg_3, avg_4, avg_5]
err_matrix = np.zeros([5,10])


for data in range(5):
    for i in range(5):
        pca = PCA(n_components = i, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
        pc = pca.fit_transform(data_list[data])
        reform = pca.inverse_transform(pc) + avg_iris
        err_matrix[data, i] = comp_err(reform, iris)

    for j in range(5):
        pca = PCA(n_components = j, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
        pc = pca.fit_transform(data_list[data])
        reform = pca.inverse_transform(pc) + avg_list[data]
        err_matrix[data, j + 5] = comp_err(reform, iris)


#reconstruction of dataset II with 2 PCs
pca = PCA(n_components = 2, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
pc = pca.fit_transform(data_2)
reform = pca.inverse_transform(pc) + avg_2
dict = {'X1': reform[:,0], 'X2':reform[:,1], 'X3':reform[:,2], 'X4':reform[:,3]}
df = pd.DataFrame(data=dict)
df.to_csv("hqiu9-recon.csv", index=False)


#Create csv for the error chart
dict2 = {'0N':err_matrix[:,0], '1N':err_matrix[:,1], '2N': err_matrix[:,2], '3N': err_matrix[:,3], '4N': err_matrix[:,4],
'0c': err_matrix[:,0], '1c':err_matrix[:,1], '2c':err_matrix[:,2], '3c':err_matrix[:,3], '4c':err_matrix[:,4]}
df2 = pd.DataFrame(data=dict2)
df2.to_csv("hqiu9-numbers.csv", index=False, header=False)
