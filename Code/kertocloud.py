# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:15:43 2017

@author: Jadon
"""


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
# Fix for Issue - #3 https://github.com/shreyans29/thesemicolon/issues/3
K.set_image_dim_ordering('th')

import numpy as np

# Image manipulations and arranging data
import os
from PIL import Image
#Sklearn to modify the data

from sklearn.cross_validation import train_test_split
#o#s.chdir("D:\semicolon\Deep Learning");

# input image dimensions
m,n = 100,100
path1="test/old/";
path2="../data/Generated/Resized/";

classes=os.listdir(path2)
x=[]
y=[]
for fol in classes:
    print (fol)
    imgfiles=os.listdir(path2+'\\'+fol);
    for img in imgfiles:
        im=Image.open(path2+'\\'+fol+'\\'+img);
        im=im.convert(mode='RGB')
        imrs=im.resize((m,n))
        imrs=img_to_array(imrs)/255;
        imrs=imrs.transpose(2,0,1);
        imrs=imrs.reshape(3,m,n);
        x.append(imrs)
        y.append(fol)
        

        
        

x=np.array(x);
y=np.array(y);

batch_size=32
nb_classes=len(classes)
nb_filters=32
nb_pool=2
nb_conv=3

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)


def train_2l_cnn(ep):
    nb_epoch=ep
    model= Sequential()
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
             
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
             
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    
    
    model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
   
    
    model_json = model.to_json()
    with open("model_2l.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_2l.h5")
    print("Saved model to disk")
    
def train_4l_cnn(ep):
    nb_epoch=ep
    model= Sequential()
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
             
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
             
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.4))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    
    
    model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
   
    
    model_json = model.to_json()
    with open("model_4l.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_4l.h5")
    print("Saved model to disk")
    
def train_5l_cnn(ep):
    nb_epoch=ep
    model= Sequential()
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
             
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
    
    model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
    model.add(Activation('relu'));
    model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
             
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    
    
    model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
   
    
    model_json = model.to_json()
    with open("model_5l.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_5l.h5")
    print("Saved model to disk")
    
if __name__ =="__main__":
    
    train_2l_cnn(1)
    train_4l_cnn(1)
    train_5l_cnn(1)
    