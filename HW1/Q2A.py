import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB

origin_image = pd.read_csv("train.csv", sep = ',', header = 0, usecols = range(1,785)) #Index from 0 to 784

#Crop all the images
from skimage.transform import resize
cropped_images = np.ones((origin_image.shape[0], 400))

for itr in range(10000):
    curr_img = origin_image.loc[[itr]].values
    img_matrix = curr_img.reshape([28,28])

    row_bounds = np.where(img_matrix > 0)[0]
    col_bounds = np.where(img_matrix > 0)[1]
    max_row_diff = max(row_bounds) - min(row_bounds)
    max_col_diff = max(col_bounds) - min(col_bounds)

    start_with_row = False
    if max_row_diff > max_col_diff:
        max_bound = max_row_diff
        start_with_row = True
    else:
        max_bound = max_col_diff


    if start_with_row:
        crop_img = img_matrix[row_bounds[0]:row_bounds[0] + max_row_diff, 0:27]
        scaled = resize(crop_img, (20, 20), preserve_range = True)
        plt.imshow(scaled, cmap = 'gray')
        cropped_images[itr] = scaled.flatten()

    else:
        crop_img = img_matrix[0 : 27, col_bounds[0]:col_bounds[0] + max_col_diff]
        scaled = resize(crop_img, (20, 20), preserve_range = True)
        plt.imshow(scaled, cmap = 'gray')
        cropped_images[itr] = scaled.flatten()
