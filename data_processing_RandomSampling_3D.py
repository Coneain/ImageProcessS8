#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:09:51 2017

Description: reservoir sampling
@author: yaoyaoyao
"""

import dicom # for reading dicom files
import os # for doing directory operations 
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
# Change this to wherever you are storing your data:
# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER


IMG_PX_SIZE = 150
HM_SLICES = 20

data_dir = './data/sample_images/'
patients = os.listdir(data_dir)
labels = pd.read_csv('./data/stage1_labels.csv', index_col=0)




IMG_SIZE_PX = 50
SLICE_COUNT = 20

def reservoir_sampling(items, k):
    """ 
    Reservoir sampling algorithm for large sample space or unknow end list
    See <http://en.wikipedia.org/wiki/Reservoir_sampling> for detail>
    Type: ([a] * Int) -> [a]
    Prev constrain: k is positive and items at least of k items
    Post constrain: the length of return array is k
    """
    sample = items[0:k]
    for i in range(k, len(items)):
        j = random.randrange(0, i-1)
        if j < k:
            temp = sample[j]
            sample[j] = items[i]
            items[i] = temp 
    return sample
    
    
def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(a):
    return sum(a) / len(a)


def process_data(patient,labels_df,img_px_size=50, hm_slices=20, visualize=False):
    
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    
    chunk_sizes = math.ceil(len(slices) / hm_slices)
    new_slices = reservoir_sampling(slices,hm_slices)
#    for slice_chunk in chunks(slices, chunk_sizes):
#        slice_chunk = list(map(mean, zip(*slice_chunk)))
#        new_slices.append(slice_chunk)
#
#    if len(new_slices) == hm_slices-1:
#        new_slices.append(new_slices[-1])
#
#    if len(new_slices) == hm_slices-2:
#        new_slices.append(new_slices[-1])
#        new_slices.append(new_slices[-1])
#
#    if len(new_slices) == hm_slices+2:
#        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
#        del new_slices[hm_slices]
#        new_slices[hm_slices-1] = new_val
#        
#    if len(new_slices) == hm_slices+1:
#        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
#        del new_slices[hm_slices]
#        new_slices[hm_slices-1] = new_val
#
#    if visualize:
#        fig = plt.figure()
#        for num,each_slice in enumerate(new_slices):
#            y = fig.add_subplot(4,5,num+1)
#            y.imshow(each_slice, cmap='gray')
#        plt.show()
        
    return np.array(new_slices),label

#                                               stage 1 for real.


x_train = []
y_train = []
for num,patient in enumerate(patients):
    if num % 100 == 0:
        print(num)
    try:
        img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        x_train.append([img_data])
        y_train.append([label])
    except KeyError as e:
        print('This is unlabeled data!')

print("Data Processing Done!")
np.save('./data/x_train_ra-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), x_train)
np.save('./data/y_train_ra-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), y_train)