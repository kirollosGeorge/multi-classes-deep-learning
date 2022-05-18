# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 03:48:05 2022

@author: Kerillos
"""
import numpy as np
import pandas as pd


from keras.preprocessing.image import load_img, img_to_array



img_width, img_height = 256, 256


def preprocess_image(path):
    img = load_img(path, target_size = (img_height, img_width))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a


test_images_dir = 'dataset/alien_test/'


test_df = pd.read_csv('dataset/test.csv')

test_dfToList = test_df['Image_id'].tolist()
test_ids = [str(item) for item in test_dfToList]



test_images = [test_images_dir+item for item in test_ids]
test_preprocessed_images = np.vstack([preprocess_image(fn) for fn in test_images])


np.save('test_preproc_CNN.npy', test_preprocessed_images)


