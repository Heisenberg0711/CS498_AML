import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#This function calculates MSE
from scipy.spatial.distance import sqeuclidean
def comp_err(A, B):
    error = 0.0
    for elemA, elemB in zip(A,B):
        error += sqeuclidean(elemA, elemB)
    return error / A.shape[0]

d1 = unpickle("cifar-10-batches-py/data_batch_1")
d2 = unpickle("cifar-10-batches-py/data_batch_2")
d3 = unpickle("cifar-10-batches-py/data_batch_3")
d4 = unpickle("cifar-10-batches-py/data_batch_4")
d5 = unpickle("cifar-10-batches-py/data_batch_5")

#Get the average image of all category of all batches and store them in dictionary.
data_batch = [d1,d2,d3,d4,d5]
img_avg = dict.fromkeys([0, 1, 2, 3, 4])
img_original = dict.fromkeys([0, 1, 2, 3, 4])
err_matrix = np.zeros([5,10])

for batch in range(5):
    imgs = {}
    imgs_mean = {}
    for n in range(10):
        keys = np.asarray(data_batch[batch][b'labels'])
        labels = np.where(keys == n)
        imgs[n] = data_batch[batch][b'data'][labels] - np.mean(data_batch[batch][b'data'][labels], axis=0)
        imgs_mean[n] = np.mean(data_batch[batch][b'data'][labels], axis=0)
    img_avg[batch] = imgs_mean
    img_original[batch] = imgs


for batch in range(5):
    for cat in range(10):
        pca = PCA(n_components = 20, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
        pc = pca.fit_transform(img_original[batch][cat])
        reform = pca.inverse_transform(pc) + img_avg[batch][cat]
        err_matrix[batch, n] = comp_err(reform, img_original[batch][cat])
