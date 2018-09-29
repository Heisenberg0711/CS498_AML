import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


d1 = unpickle("cifar-10-batches-py/data_batch_1")
d2 = unpickle("cifar-10-batches-py/data_batch_2")
d3 = unpickle("cifar-10-batches-py/data_batch_3")
d4 = unpickle("cifar-10-batches-py/data_batch_4")
d5 = unpickle("cifar-10-batches-py/data_batch_5")

data_batch = [d1,d2,d3,d4,d5]
img_avg = dict.fromkeys([0, 1, 2, 3, 4])

for batch in range(5):
    imgs = {}
    for n in range(10):
        keys = np.asarray(d1[b'labels'])
        labels = np.where(k1 == n)
        img_n = np.mean(d1[b'data'][labels], axis=0)
        imgs[n] = img_n
    img_avg[batch] = imgs
