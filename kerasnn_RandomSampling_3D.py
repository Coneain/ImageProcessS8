#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:12:04 2017

Description: reservoir sampling
@author: yaoyaoyao
"""


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

IMG_SIZE_PX = 50
SLICE_COUNT = 20
x_train = np.load("./data/x_train_ra-"+str(IMG_SIZE_PX)+"-"+str(IMG_SIZE_PX)+"-20.npy")
y_train = np.load("./data/y_train_ra-"+str(IMG_SIZE_PX)+"-"+str(IMG_SIZE_PX)+"-20.npy")

# CNN Training parameters



n_classes = 2
batch_size = 10
nb_epoch = 10

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, n_classes)


# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]

# Pre-processing
train_set = np.array(x_train)
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)



# Define model

model = Sequential()
model.add(Convolution3D(32,5,5,5, 
                        input_shape=(1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX,), 
                                    activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(n_classes,init='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])

  
# Split the data

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)


# Train the model

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
          batch_size=batch_size,nb_epoch = nb_epoch,shuffle=True)

 # Evaluate the model
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('Test score:', score[0])
print('Accuracy :', score[1])

