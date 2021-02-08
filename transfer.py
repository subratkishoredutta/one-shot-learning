# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:24:27 2021

@author: Asus
"""
import os
import cv2
from  tqdm import tqdm
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import keras

def siamese():
  base_input=Input((224,224,3))
  test_input=Input((224,224,3))
  model=Sequential()
  vgg=VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
  for layer in vgg.layers:
      layer.trainable = False
  model.add(vgg)
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  
  Aencod=model(base_input)
  
  Tencod=model(test_input)
  
  dLayer=Lambda(lambda tensor:K.abs(tensor[0]-tensor[1]))
  
  ldist=dLayer([Aencod,Tencod])
  
  output=Dense(1,activation="sigmoid")(ldist)
  
  network=Model(inputs=[base_input,test_input],outputs=output)
  return (network,model)       


model,mdl=siamese()
model.summary()
posAnchor='D:/coursera/deep learning specialisation/siamese/pos/Anchor/'
negAnchor='D:/coursera/deep learning specialisation/siamese/negative/neganchor/'

posref='D:/coursera/deep learning specialisation/siamese/pos/Reference/'
negref='D:/coursera/deep learning specialisation/siamese/negative/negReference/'
SIZE=20000
APimages=np.zeros((SIZE, 224, 224,3),dtype=np.uint8)
RPimages=np.zeros((SIZE, 224, 224,3),dtype=np.uint8)

ANimages=np.zeros((SIZE, 224, 224,3),dtype=np.uint8)
RNimages=np.zeros((SIZE, 224, 224,3),dtype=np.uint8)

i=0
for Aname,Rname in tqdm(zip(os.listdir(posAnchor),os.listdir(posref))):
    if Aname==Rname and i<SIZE:
        Apath=os.path.join(posAnchor,Aname)
        Rpath=os.path.join(posref,Rname)
        Aimg=cv2.resize(cv2.imread(Apath),(224,224))
        Rimg=cv2.resize(cv2.imread(Rpath),(224,224))
        APimages[i]=Aimg
        RPimages[i]=Rimg
        i+=1
i=0
for Aname,Rname in tqdm(zip(os.listdir(negAnchor),os.listdir(negref))):
    if Aname==Rname and i<SIZE:
        Apath=os.path.join(negAnchor,Aname)
        Rpath=os.path.join(negref,Rname)
        Aimg=cv2.resize(cv2.imread(Apath),(224,224))
        Rimg=cv2.resize(cv2.imread(Rpath),(224,224))
        ANimages[i]=Aimg
        RNimages[i]=Rimg
        i+=1

XA=np.zeros((len(APimages[:len(ANimages)])+len(ANimages),224,224,3),dtype=np.uint8)
XR=np.zeros((len(RPimages[:len(RNimages)])+len(RNimages),224,224,3),dtype=np.uint8)
Y = np.ones((2*len(ANimages)),dtype=np.uint8)
i=0
for  AP,AN,RP,RN in tqdm(zip(APimages[:len(ANimages)],ANimages,RPimages[:len(RNimages)],RNimages)):
   Y[i]=0
   XA[i] = AP
   XA[i+1] = AN
   XR[i] = RP
   XR[i+1] = RN
   i+=2

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([XA,XR],Y,validation_split=0.1,epochs=20,batch_size=32)
model.save('siamese.h5')
