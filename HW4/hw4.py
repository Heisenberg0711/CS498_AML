import numpy as np
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
err_matrix = np.zeros(10)
#This matrix is used to store the representations of 10 categories
Principles = np.zeros([10,20,3072])


#Calculate MSE of each category of images
for cat in range(10):
    pca = PCA(n_components = 20, copy=True, whiten=False, svd_solver='full', iterated_power='auto')
    pc = pca.fit_transform(train_data[train_label == cat,:])
    reform = pca.inverse_transform(pc)
    err_matrix[cat] = comp_err(reform, train_data[train_label == cat,:])
    Principles[cat] = pca.components_

plt.figure(1,[10, 5])
plt.bar(range(0,10), err_matrix, align = 'center', alpha = 0.9)
plt.title('MSE of categories 0 to 9', fontsize = 15)
plt.xlabel('Categories')
plt.ylabel('Mean Squared Error')
plt.xticks(range(10),LabelNames)
plt.show()



#Problem 7.7(b)
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
#Set the random seed to control the MDS function
seed = np.random.RandomState(seed=3)
pcoa = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
           dissimilarity="precomputed", n_jobs=1)

distances = euclidean_distances(img_avg)
pos = pcoa.fit(distances).embedding_

plt.figure(figsize=(8,6))
plt.scatter(pos[:,0], pos[:,1], color='turquoise')
plt.title('2D Map Plot of All Categories of Images', fontsize = 15)
for i in range(10):
    plt.annotate(LabelNames[i], pos[i,:])
