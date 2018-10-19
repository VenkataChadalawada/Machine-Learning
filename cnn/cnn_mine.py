#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:23:03 2018

@author: vchadalawada
"""


# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 0 - Data Preprocesiing - first we seprate cats and dogs folder for both training and testing data to simplify for keras
# we do this preparation manually and created folders


# Part 1 - Building comvolutional neural network
from keras.models import Sequential # to initialize a NN (we can do as a graph or sequential)
from keras.layers import Convolution2D # since images are 2D we use convolution2D
from keras.layers import MaxPooling2D # for pooling step
from keras.layers import Flatten # for flatten
from keras.layers import Dense # to add fully connection layers(hidden layers for CNN)

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution applying sevral feature detectors to bring feature maps
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu')) #no of feature detector = 32 matrices with 3 X 3
# input shape for color image is 3d so it would be 3X3X256X256 as it would be 256 X 256 images

# Step 2 - Pooling
# reducing size of feature map using a pooling technique (maxpooling we chosen) will bring pooled feature map
classifier.add(MaxPooling2D(pool_size = (2, 2))) # 2X2 is size that gets chunked and totaled to position the sum in pooledfeatured map

# Adding a seconf convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) #no of feature detector = 32 matrices with 3 X 3
classifier.add(MaxPooling2D(pool_size = (2, 2))) # 2X2 is size that gets chunked and totaled to position the sum in pooledfeatured map
# you dont need input_shape as keras will already know it

# Step 3 - Flattening
# from pooling layer faltten the data into flat 1D layer of inputs
classifier.add(Flatten())

# Step4 - Full connection
# output_dim => no of nodes in hidden layer= we ended up lot of input nodes so we cant simply avg them out, by trial end error we chose 128
classifier.add(Dense(output_dim = 128, activation='relu'))
# output layer
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#step5 - compiling
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#stochastic gradient => adam

#Part 2 - Fitting the CNN to the images
# image augumentation - preprocessing images to ensure overfitting problem - which is great result on training step but not in testing step
# keras has supporting function ImageDataGenerator
# when we have few data - > our model fail to generate correlations -> it requires large data set 
# trick is IMAGE AUGUMENTATION -> SInce we dont have lots and lots of data -> It selects randomly a subset of our same data set to create data sets by rotating, flipping, shifting them
# this trick helps solving overfitting problem

#rescale
#shear_range -> geometrical transformation
#zoom_range
#horizontal_flip
 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
#target size is an expected image sizes, batch size =>after 32 images it goes into CNN to apply the backpropogation weight change, 

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
# samples_per_epoch => no of images in training set, nb_epoch = no of epochs to train our neural network

### Now accuracy of the model

#To improve we need to make more deeper layers
# 1) adding convolutional layer (includes another maxpooling step) (or)
# 2) adding hidden layer