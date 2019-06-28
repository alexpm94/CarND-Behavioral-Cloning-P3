
# coding: utf-8

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DataSet():
    def __init__(self,PATH):
        self.X = []
        self.Y = []
        self.PATH = PATH
        self.df = pd.read_csv(PATH+'driving_log.csv',names = ['Center', 'Left', 'Right', 's_c','s_l','s_r','speed'])
        self.IMAGE_PATH = PATH+'IMG/'
        self.set_dataSet()
    
    def get_ListOfImages(self, side):
        images = []
        for path in self.df[side]:
            current_path = self.IMAGE_PATH + path.split('/')[-1]
            image = cv2.imread(current_path)
            images.append(image)
        return images

    def get_steerings(self,side, correction):
        return np.array(self.df[side], dtype ='float32')+correction
    
    def get_X(self):
        print('Loading Images..... ')
        images_center = self.get_ListOfImages('Center')
        images_left   = self.get_ListOfImages('Left')
        images_right  = self.get_ListOfImages('Right')
        self.X = np.array(images_center+images_left+images_right)
        print('Images loaded')

    def get_Y(self):
        steering_center = self.get_steerings('s_c',0)
        steering_left   = self.get_steerings('s_c',0.1)
        steering_right  = self.get_steerings('s_c',-0.1)
        self.Y = np.concatenate([steering_center,steering_left,steering_right])

    def set_dataSet(self):
        self.get_X()
        self.get_Y()
        print('Shape of X_train: {}, y_train: {} '.format(self.X.shape,self.Y.shape))

#Test Class from a normal drving
normal_driving = DataSet('../data/data_sim/')

#Test Class from a counter wise drving
counterwise_driving = DataSet('../data/data_sim_back/')

#Test Class from a different recovering scenarios
recovering_driving = DataSet('../data/data_recover/')

#Merge the 3 datasets
print('Merging Datasets')
X_train = np.concatenate([normal_driving.X,counterwise_driving.X,recovering_driving.X])
y_train = np.concatenate([normal_driving.Y,counterwise_driving.Y,recovering_driving.Y])
print('Datasets merged')

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Shuffle images and split to get a validation set
X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=23)

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

#Create Model
batch_size = 150
epochs = 15
pool_size = (2, 2)
input_shape = X_train.shape[1:]

print('Initializing Model')
model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
# Convolutional Layer 1 and Dropout
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# Conv Layer 2
model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
# Conv Layer 3
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
# Conv Layer 4
model.add(Convolution2D(8, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=pool_size))
# Flatten and Dropout
model.add(Flatten())
model.add(Dropout(0.5))
# Fully Connected Layer 1 and Dropout
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# FC Layer 2
model.add(Dense(64))
model.add(Activation('relu'))
# FC Layer 3
model.add(Dense(32))
model.add(Activation('relu'))
# Final FC Layer - just one output - steering angle
model.add(Dense(1))
print('Model Initialized')

# Compiling and training the model
#model.compile(metrics=['mean_squared_error'], optimizer='Nadam', loss='mean_squared_error')
print('Training started ....')
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2, validation_data=(X_val, y_val))
print('Training finished')

model.save('model.h5')

# Show summary of model
model.summary()
