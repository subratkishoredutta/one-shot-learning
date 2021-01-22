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

Y1= np.ones(APimages.shape[0],dtype=np.uint8)
Y2=np.zeros(ANimages.shape[0],dtype=np.uint8)
Y=np.concatenate((Y1,Y2),axis=None)
