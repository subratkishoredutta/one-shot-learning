# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:57:53 2021

@author: Asus
"""
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,BatchNormalization,Input,Lambda  

##creating the training data 
posAnchor='D:/coursera/deep learning specialisation/siamese/pos/Anchor/'
negAnchor='D:/coursera/deep learning specialisation/siamese/negative/neganchor/'

posref='D:/coursera/deep learning specialisation/siamese/pos/Reference/'
negref='D:/coursera/deep learning specialisation/siamese/negative/negReference/'

APimages=np.zeros((len(os.listdir(posAnchor)), 128, 128,3),dtype=np.uint8)
RPimages=np.zeros((len(os.listdir(posref)), 128, 128,3),dtype=np.uint8)

ANimages=np.zeros((len(os.listdir(negAnchor)), 128, 128,3),dtype=np.uint8)
RNimages=np.zeros((len(os.listdir(negref)), 128, 128,3),dtype=np.uint8)

i=0
for Aname,Rname in tqdm(zip(os.listdir(posAnchor),os.listdir(posref))):
    if Aname==Rname:
        Apath=os.path.join(posAnchor,Aname)
        Rpath=os.path.join(posref,Rname)
        Aimg=cv2.imread(Apath)
        Rimg=cv2.imread(Rpath)
        APimages[i]=Aimg
        RPimages[i]=Rimg
        i+=1
i=0
for Aname,Rname in tqdm(zip(os.listdir(negAnchor),os.listdir(negref))):
    if Aname==Rname:
        Apath=os.path.join(negAnchor,Aname)
        Rpath=os.path.join(negref,Rname)
        Aimg=cv2.imread(Apath)
        Rimg=cv2.imread(Rpath)
        ANimages[i]=Aimg
        RNimages[i]=Rimg
        i+=1

XA=np.concatenate((APimages,ANimages),axis=0)##anchor images
XR=np.concatenate((RPimages,RNimages),axis=0)##reference images
Y1= np.ones(APimages.shape[0],dtype=np.uint8)#labels for images with same person
Y2=np.zeros(ANimages.shape[0],dtype=np.uint8)#label for images with different person
Y=np.concatenate((Y1,Y2),axis=None)

#model declaration
     
        
def siamese(input_shape):
  base_input=Input(input_shape)
  test_input=Input(input_shape)

  model=Sequential()
  model.add(Conv2D(5,(3,3),padding='same',activation="relu",kernel_initializer="he_normal"))
  model.add(Conv2D(25,(3,3),padding='same',activation="relu",kernel_initializer="he_normal"))
  model.add(MaxPool2D((2,2),padding='same'))
  model.add(Conv2D(125,(3,3),padding="same",activation="relu",kernel_initializer="he_normal"))
  model.add(Flatten())
  model.add(Dense(128,activation="relu",name="OUT"))

  Aencod=model(base_input)
  Tencod=model(test_input)

  dLayer=Lambda(lambda tensor:K.abs(tensor[0]-tensor[1]))
  ldist=dLayer([Aencod,Tencod])
  output=Dense(1,activation="sigmoid")(ldist)
  network=Model(inputs=[base_input,test_input],outputs=output)
  return (network,model)       
        
model,mdl=siamese((128,128,3))      
        
model.compile(loss="binary_crossentropy",optimizer="Adam",metrics=["accuracy"])

#model training
model.fit([XA,XR],[Y],epochs=10)
