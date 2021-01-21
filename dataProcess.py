# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:49:55 2021

@author: Asus
"""

import os
from tqdm import tqdm
path='D:/coursera/deep learning specialisation/siamese/vggface2_test/test'
import cv2
import numpy as np
import matplotlib.pyplot as plt
##data prep

##positives
anchorpath='D:/coursera/deep learning specialisation/siamese/pos/Anchor/'
refpath='D:/coursera/deep learning specialisation/siamese/pos/Reference/'


name=0
for folders in tqdm(os.listdir(path)):
    reqPath=os.path.join(path,folders)
    l=int(len(os.listdir(reqPath))/2)
    for i in range(int(len(os.listdir(reqPath))/2)):
        length=i+l
        Aname=os.listdir(reqPath)[i]
        Rname=os.listdir(reqPath)[length]
        aPATH=os.path.join(reqPath,Aname)
        rPATH=os.path.join(reqPath,Rname)
        Aimg=cv2.resize(cv2.imread(aPATH),(128,128))
        Rimg=cv2.resize(cv2.imread(rPATH),(128,128))
        if np.sum(Aimg)!=None and np.sum(Rimg)!= None:
            Aname=anchorpath+str(name)+".jpeg"
            Rname=refpath+str(name)+".jpeg"
            name+=1
            
            cv2.imwrite(Aname,Aimg)
            cv2.imwrite(Rname,Rimg)
            
##negatives
neganchorpath='D:/coursera/deep learning specialisation/siamese/negative/anchor/'
negrefpath= 'D:/coursera/deep learning specialisation/siamese/negative/Reference/'

for folders in tqdm(os.listdir(path)):
    reqpath=os.path.join(path,folders)
    o=0
    for i in range(int(len(os.listdir(reqpath))/2)):
        Aname=os.listdir(reqpath)[i]
        rpath=os.path.join(path,os.listdir(path)[i])
        Rname=os.listdir(rpath)[0]
        Apath=os.path.join(reqpath,Aname)
        Rpath=os.path.join(str(rpath),str(Rname))
        Aimg=cv2.resize(cv2.imread(Apath),(128,128))
        Rimg=cv2.resize(cv2.imread(Rpath),(128,128))
        if np.sum(Aimg)!=None and np.sum(Rimg)!= None and rpath!=reqpath:
            Aname=neganchorpath+str(name)+".jpeg"
            Rname=negrefpath+str(name)+".jpeg"
            name+=1
            
            cv2.imwrite(Aname,Aimg)
            cv2.imwrite(Rname,Rimg)
 