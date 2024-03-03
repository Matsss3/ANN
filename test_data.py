# import os
# import urllib
# import urllib.request
# from zipfile import ZipFile

# URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
# FILE = 'fashion_mnist_images.zip'
# FOLDER = 'fashion_mnist_images'

# if not os.path.isfile(FILE):
#     print(f'Downloading {URL} and saving as {FILE}...')
#     urllib.request.urlretrieve(URL, FILE)
    
# print('Unzipping...')
# with ZipFile(FILE) as zip_images:
#     zip_images.extractall(FOLDER)
    
# print('Done!')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d0 = pd.read_csv('./mnist_dataset/train.csv')

l = d0['label']

d = d0.drop('label', axis=1)

d_data = d.values

X = (d_data.reshape(d_data.shape[0], -1).astype(np.float32) - 127.5) / 127.5

print(X[:5])