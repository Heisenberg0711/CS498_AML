import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Problem 7.7 (a)
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
img_center = dict.fromkeys([0, 1, 2, 3, 4])
img_avg = dict.fromkeys([0, 1, 2, 3, 4])
img_original = dict.fromkeys([0, 1, 2, 3, 4])
err_matrix = np.zeros([5,10])

for batch in range(5):
    imgs_c = {}
    imgs_mean = {}
    img_org = {}
    for n in range(10):
        keys = np.asarray(data_batch[batch][b'labels'])
        labels = np.where(keys == n)
        img_org[n] = data_batch[batch][b'data'][labels]
        imgs_c[n] = data_batch[batch][b'data'][labels] - np.mean(data_batch[batch][b'data'][labels], axis=0)
        imgs_mean[n] = np.mean(data_batch[batch][b'data'][labels], axis=0)
    img_center[batch] = imgs_c
    img_avg[batch] = imgs_mean
    img_original[batch] = img_org


for batch in range(5):
    for n in range(10):
        pca = PCA(n_components = 20, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
        pc = pca.fit_transform(img_center[batch][n])
        reform = pca.inverse_transform(pc) + img_avg[batch][n]
        err_matrix[batch, n] = comp_err(reform, img_original[batch][n])

plt.bar(range(0,10), err_matrix[0])
plt.title('MSE of categories 0 to 9')
plt.xlabel('Categories')
plt.ylabel('Mean Squared Error')
plt.show()



#Problem 7.7(b)
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
seed = np.random.RandomState(seed=5)
pcoa = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
           dissimilarity="precomputed", n_jobs=1)

a = img_avg[1][0]
for idx in range(1,10):
    b = img_avg[1][idx]
    a = np.concatenate((a,b), axis=0)
a = a.reshape(10,3072)
print(a.shape)

distances = euclidean_distances(a)
pos = pcoa.fit(distances).embedding_
plt.scatter(pos[:,0], pos[:,1], color='turquoise', lw=0, label='pcoa')
plt.show()
