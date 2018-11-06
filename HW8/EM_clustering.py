import numpy as np
from scipy import misc
from scipy.spatial import distance
from scipy.cluster.vq import kmeans

def CloseCluster(pixel, p, nclusters):
    pixelArray = np.tile(pixel,(nclusters,1))
    dist = distance.cdist(pixelArray, p, 'euclidean')
    return np.argmin(dist)
