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

meta_data = unpickle("cifar-10-batches-py/batches.meta")
origin_train_data = [unpickle("cifar-10-batches-py/data_batch_1"),
              unpickle("cifar-10-batches-py/data_batch_2"),
              unpickle("cifar-10-batches-py/data_batch_3"),
              unpickle("cifar-10-batches-py/data_batch_4"),
              unpickle("cifar-10-batches-py/data_batch_5")]
origin_test_data = unpickle("cifar-10-batches-py/test_batch")

#Get the average image of all category of all batches and store them in dictionary.
train_data = np.array([x[b'data'] for x in origin_train_data]).reshape(50000, 3072)
train_label = np.array([x[b'labels'] for x in origin_train_data]).reshape(50000)
test_data = origin_test_data[b'data']   #Test data has shape of (10000,3072)
test_label = np.array(origin_test_data[b'labels'])

#Sift images based on label and calculate the average of images (for part b)
img_avg = np.array([np.mean(train_data[train_label == label,], axis=0) for label in range(10)])
#Get names for each numerical category from the meta data
LabelNames = np.array([str(name,encoding='utf-8') for name in meta_data[b'label_names']])
err_matrix = np.zeros([1,10])
#This matrix is used to store the representations of 10 categories
Principles = np.zeros([10,20,3027])



# for batch in range(5):
#     imgs_c = {}
#     imgs_mean = {}
#     img_org = {}
#     for n in range(10):
#         keys = np.asarray(data_batch[batch][b'labels'])
#         labels = np.where(keys == n)
#         img_org[n] = data_batch[batch][b'data'][labels]
#         imgs_c[n] = data_batch[batch][b'data'][labels] - np.mean(data_batch[batch][b'data'][labels], axis=0)
#         imgs_mean[n] = np.mean(data_batch[batch][b'data'][labels], axis=0)
#     img_center[batch] = imgs_c
#     img_avg[batch] = imgs_mean
#     img_original[batch] = img_org

#Calculate MSE of each category of images
for category in range(10):
    pca = PCA(n_components = 20, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
    pc = pca.fit_transform(train_data[train_label == category])
    reform = pca.inverse_transform(pc)
    err_matrix[category] = comp_err(reform, train_data[train_label == category])
    Principles[category] = pca.components_

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
