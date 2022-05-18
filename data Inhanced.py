# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 03:49:09 2022

@author: Kerillos
"""

import os
import random
from shutil import copyfile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
import pathlib

# Data Analysis of data 
image_folder = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
nimgs = {}
for i in image_folder:
    nimages = len(os.listdir('dataset/'+i+'/'))
    nimgs[i]=nimages
plt.figure(figsize=(10, 8))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes of Dataset')
plt.show()

try:
    os.mkdir('weather_pred/Data')
    os.mkdir('weather_pred/Data/training')
    os.mkdir('weather_pred/Data/validation')
    os.mkdir('weather_pred/Data/training/cloudy')
    os.mkdir('weather_pred/Data/training/foggy')
    os.mkdir('weather_pred/Data/training/rainy')
    os.mkdir('weather_pred/Data/training/shine')
    os.mkdir('weather_pred/Data/training/sunrise')
    os.mkdir('weather_pred/Data/validation/cloudy')
    os.mkdir('weather_pred/Data/validation/foggy')
    os.mkdir('weather_pred/Data/validation/rainy')
    os.mkdir('weather_pred/Data/validation/shine')
    os.mkdir('weather_pred/Data/validation/sunrise')
except OSError:
    pass
# Function to split data to train and validation 
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    valid_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    valid_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in valid_set:
        this_file = SOURCE + filename
        destination = VALIDATION + filename
        copyfile(this_file, destination)
        
# Dir of all data 
CLOUDY_SOURCE_DIR = 'dataset/cloudy/'
TRAINING_CLOUDY_DIR = 'weather_pred/Data/training/cloudy/'
VALID_CLOUDY_DIR = 'weather_pred/Data/validation/cloudy/'

FOGGY_SOURCE_DIR = 'dataset/foggy/'
TRAINING_FOGGY_DIR = 'weather_pred/Data/training/foggy/'
VALID_FOGGY_DIR = 'weather_pred/Data/validation/foggy/'

RAINY_SOURCE_DIR = 'dataset/rainy/'
TRAINING_RAINY_DIR = 'weather_pred/Data/training/rainy/'
VALID_RAINY_DIR = 'weather_pred/Data/validation/rainy/'

SHINE_SOURCE_DIR = 'dataset/shine/'
TRAINING_SHINE_DIR = 'weather_pred/Data/training/shine/'
VALID_SHINE_DIR = 'weather_pred/Data/validation/shine/'

SUNRISE_SOURCE_DIR = 'dataset/sunrise/'
TRAINING_SUNRISE_DIR = 'weather_pred/Data/training/sunrise/'
VALID_SUNRISE_DIR = 'weather_pred/Data/validation/sunrise/'

# split 85% train data and 15% test data
split_size = .85

split_data(CLOUDY_SOURCE_DIR, TRAINING_CLOUDY_DIR, VALID_CLOUDY_DIR, split_size)
split_data(FOGGY_SOURCE_DIR, TRAINING_FOGGY_DIR, VALID_FOGGY_DIR, split_size)
split_data(RAINY_SOURCE_DIR, TRAINING_RAINY_DIR, VALID_RAINY_DIR, split_size)
split_data(SHINE_SOURCE_DIR, TRAINING_SHINE_DIR, VALID_SHINE_DIR, split_size)
split_data(SUNRISE_SOURCE_DIR, TRAINING_SUNRISE_DIR, VALID_SUNRISE_DIR, split_size)

#Data Analysis of train data
image_folder = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
nimgs = {}
for i in image_folder:
    nimages = len(os.listdir('weather_pred/Data/training/'+i+'/'))
    nimgs[i]=nimages
plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Training Dataset')
plt.show()

# Number of all image in train data at specific category
for i in ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']:
    print('Training {} images are: '.format(i)+str(len(os.listdir('weather_pred/Data/training/'+i+'/'))))
    
    
#Data Analysis of validation data
image_folder = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
nimgs = {}
for i in image_folder:
    nimages = len(os.listdir('weather_pred/Data/validation/'+i+'/'))
    nimgs[i]=nimages
plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Validation Dataset')
plt.show()

# Number of all image in train data at specific category
for i in ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']:
    print('Valid {} images are: '.format(i)+str(len(os.listdir('weather_pred/Data/validation/'+i+'/')))) 


