# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:57:53 2021

@author: Asus
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

posAnchor='D:/coursera/deep learning specialisation/siamese/pos/Anchor/'
negAnchor='D:/coursera/deep learning specialisation/siamese/negative/neganchor/'

posref='D:/coursera/deep learning specialisation/siamese/pos/Reference/'
negref='D:/coursera/deep learning specialisation/siamese/negative/negReference/'

APimages=np.zeros((80000, 128, 128,3),dtype=np.uint8)
RPimages=np.zeros((80000, 128, 128,3),dtype=np.uint8)

ANimages=np.zeros((80000, 128, 128,3),dtype=np.uint8)
RNimages=np.zeros((80000, 128, 128,3),dtype=np.uint8)

i=0
for Aname,Rname in tqdm(zip(os.listdir(posAnchor),os.listdir(posref))):
    if Aname==Rname and i<80000:
        Apath=os.path.join(posAnchor,Aname)
        Rpath=os.path.join(posref,Rname)
        Aimg=cv2.imread(Apath)
        Rimg=cv2.imread(Rpath)
        APimages[i]=Aimg
        RPimages[i]=Rimg
        i+=1
i=0
for Aname,Rname in tqdm(zip(os.listdir(negAnchor),os.listdir(negref))):
    if Aname==Rname and i<80000:
        Apath=os.path.join(negAnchor,Aname)
        Rpath=os.path.join(negref,Rname)
        Aimg=cv2.imread(Apath)
        Rimg=cv2.imread(Rpath)
        ANimages[i]=Aimg
        RNimages[i]=Rimg
        i+=1

XA=np.zeros((len(APimages[:len(ANimages)])+len(ANimages),128,128,3),dtype=np.uint8)
XR=np.zeros((len(RPimages[:len(RNimages)])+len(RNimages),128,128,3),dtype=np.uint8)
Y = np.zeros((2*len(ANimages)),dtype=np.uint8)
i=0
for  AP,AN,RP,RN in tqdm(zip(APimages[:len(ANimages)],ANimages,RPimages[:len(RNimages)],RNimages)):
   Y[i]=1
   XA[i] = AP
   XA[i+1] = AN
   XR[i] = RP
   XR[i+1] = RN
   i+=2
   
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,BatchNormalization,Input,Lambda       
        
def siamese(input_shape):
  base_input=Input(input_shape)
  test_input=Input(input_shape)

  model=Sequential()
  model.add(Conv2D(64,(3,3),padding='same',activation="relu",kernel_initializer="he_normal"))
  model.add(Conv2D(64,(3,3),padding='same',activation="relu",kernel_initializer="he_normal"))
  model.add(MaxPool2D((2,2),padding='same'))
  model.add(Conv2D(64,(3,3),padding="same",activation="relu",kernel_initializer="he_normal"))
  model.add(Flatten())
  model.add(Dense(50,activation="relu",name="OUT"))

  Aencod=model(base_input)
  Tencod=model(test_input)

  dLayer=Lambda(lambda tensor:K.abs(tensor[0]-tensor[1]))
  ldist=dLayer([Aencod,Tencod])
  output=Dense(1,activation="sigmoid")(ldist)
  network=Model(inputs=[base_input,test_input],outputs=output)
  return (network,model)       
        
model,mdl=siamese((128,128,3))      
        
model.compile(loss="binary_crossentropy",optimizer="Adam",metrics=["accuracy"])


model.fit([XA[:30000],XR[:30000]],[Y[:30000]],validation_split = 0.1,batch_size=64,epochs=1000)


model.save('siamese110.h5')
