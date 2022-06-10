#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:35:24 2022

@author: barbereaup
"""


import numpy as np
import os
from PIL import Image
import torch
import data_visualisation as dv
import matplotlib.pyplot as plt
import torch.nn as nn


#name_list=os.listdir(path)
def dataload(path500,path1200):
    name_list=os.listdir(path500)
    name_list.sort()
    output500=[]
    output1200=[]
    name_list_test=[]
    for im in name_list:
        im500 = None
        im1200 = None
        name_list_test.append(im.split(".")[0].split("reunion")[0]+"indien0025.npy")
        
        file1 = f"%s/%s"%(path500,im)
        if os.path.exists(file1):
            im500 = np.load(file1)
        else:
            print("[ERROR] File does not exist: " + str(file1))
            continue
        
        file2 = f"%s/%s"%(path1200,im.split(".")[0].split("reunion")[0]+"indien0025.npy")
        if os.path.exists(file2):
            im1200 = np.load(file2)
        else:
            print("[ERROR] File does not exist: " + str(file2))
            continue
           
        print("[INFO] Add the file: " + str(file1))
        output500.append(im500)
        output1200.append(im1200)
        
    
#        print(f"%s/%s"%(path500,im))
#        print(im)
#        print(im.split(".")[0].split("reunion")[0]+"indien0025.npy")
    return (np.asarray(output500),np.asarray(output1200),name_list_test)

def fulldataload(variables):
    name_list=os.listdir("../data500/%s"%variables[0])
    name_list.sort()
    output500=[]
    output1200=[]
    name_list_test=[]
    for im in name_list:
        name_list_test.append(im.split(".")[0].split("reunion")[0]+"indien0025")
        data500=[]
        data1200=[]
        for var in variables:
            im500 = None
            im1200 = None
            path500=f"../data500/%s"%var
            path1200=f"../data1200/%s"%var
            
            file1 = f"%s/%s"%(path500,im.split("_")[0]+"_"+var+"_"+im.split("_")[2]+"_"+im.split("_")[3])
            if os.path.exists(file1):
                im500 = np.load(file1)
            else:
                print("[ERROR] File does not exist: " + str(file1))
                continue
            
            file2 = f"%s/%s"%(path1200,(im.split("_")[0]+"_"+var+"_"+im.split("_")[2]+"_"+im.split("_")[3]).split(".")[0].split("reunion")[0]+"indien0025.npy")
            if os.path.exists(file2):
                im1200 = np.load(file2)
            else:
                print("[ERROR] File does not exist: " + str(file2))
                continue
               
            print("[INFO] Add the file: " + str(file1))
            data500.append(im500)
            data1200.append(im1200)
        output500.append(data500)
        output1200.append(data1200)
    return np.asarray(output500),np.asarray(output1200),name_list_test

def train_test_split(X,Y,name_list):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    name_list_test=[]
    i=0
    for x,y,name in zip(X,Y,name_list):
        if 0<=i<20:
            X_train.append(x)
            Y_train.append(y)
        if 21<=i<=25:
            X_test.append(x)
            Y_test.append(y)
            name_list_test.append(name)
        if i == 26:
            i=-1
        i+=1
    return np.asarray(X_train),np.asarray(X_test),np.asarray(Y_train),np.asarray(Y_test),name_list_test
        
    

def jpeg_load_meteo(path,var='temp850'):
    #Temp850 only
    name_list=os.listdir(path)
    name_list.sort()
    array_list=[]
    original=[]
    interp=[]
    pred=[]
    target=[]
    output=[]
    visu=[]
    for name in name_list:
        if name.split(".")[-1]=="png":
            array_list.append(np.array(Image.open(f'%s/%s'%(path,name))))
            splitname=name.split("_")
            original.append(np.array(Image.open(f'../data1200/%s/%s.png'%(var,name.split("-")[0])))[:256,:256,:])
#            print("torch.from_numpy(original[-1]).unsqueeze(0).float().permute(0,3,1,2).shape = ",\
#                  torch.from_numpy(original[-1]).unsqueeze(0).float().permute(0,3,1,2).shape)
            inter = torch.from_numpy(original[-1]).unsqueeze(0).float().permute(0,3,1,2)
#            print (" inter 1 = ", inter.shape)
            inter = nn.functional.interpolate(inter,scale_factor=2,mode='bicubic')
#            print("inter2 = ",inter.shape)
            inter = inter.permute(0,2,3,1).detach().numpy().astype(np.uint8)[0]
#            print("inter.shape = ",inter.shape)
            interp.append(inter)
    for im in array_list:
        pred.append(im[:512,:512,:])
        target.append(im[:,512:,:])
#    print("original[0].shape = ",original[0].shape)
#    print("interp[0].shape = ",interp[0].shape)
#    print("pred[0].shape = ",pred[0].shape)
#    print("target[0].shape = ",target[0].shape)
    for o,i,p,t in zip(original,interp,pred,target):
        origin_rescaled=np.zeros((512,512,3))
        origin_rescaled[128:384,128:384,:]=o
        output.append([origin_rescaled,i,p,t])
    output = np.array(output).astype(np.uint8)
    
#    for im,name in zip(output,name_list):
#        
#        im=dv.visu(im)
#        visu.append(im)
##        print("im.shape = ",im.shape)
#        im=Image.fromarray(im,mode="RGB")
#
#        im.save(f'%s/%s-visu.png'%(path,name.split(".")[0].split("-")[0]))
    
    return output


def jpeg_load_images(path):
    name_list=os.listdir(path)
    name_list.sort()
    array_list=[]
    original=[]
    interp=[]
    pred=[]
    target=[]
    output=[]

    for name in name_list:
        if name.split(".")[-1]=="jpeg":
            array_list.append(np.array(Image.open(f'%s/%s'%(path,name))))
            original.append(np.array(Image.open(f'../DIV2K_train_LR_unknown/X2/%sx2.png'%(name.split("-")[0])))[:256,:256,:])
#            print("torch.from_numpy(original[-1]).unsqueeze(0).float().permute(0,3,1,2).shape = ",\
#                  torch.from_numpy(original[-1]).unsqueeze(0).float().permute(0,3,1,2).shape)
            inter = torch.from_numpy(original[-1]).unsqueeze(0).float().permute(0,3,1,2)
#            print (" inter 1 = ", inter.shape)
            inter = nn.functional.interpolate(inter,scale_factor=2,mode='bicubic')
#            print("inter2 = ",inter.shape)
            inter = inter.permute(0,2,3,1).detach().numpy().astype(np.uint8)[0]
#            print("inter.shape = ",inter.shape)
            interp.append(inter)
    for im in array_list:
        pred.append(im[:512,:512,:])
        target.append(im[:,512:,:])
#    print("original[0].shape = ",original[0].shape)
#    print("interp[0].shape = ",interp[0].shape)
#    print("pred[0].shape = ",pred[0].shape)
#    print("target[0].shape = ",target[0].shape)
    for o,i,p,t in zip(original,interp,pred,target):
        origin_rescaled=np.zeros((512,512,3))
        origin_rescaled[128:384,128:384,:]=o
        output.append([origin_rescaled,i,p,t])
    output = np.array(output).astype(np.uint8)
#    for im,name in zip(output,name_list):
#        
#        im=dv.visu(im)
#        visu.append(im)
##        print("im.shape = ",im.shape)
#        im=Image.fromarray(im,mode="RGB")
#
#        im.save(f'%s/%s-visu.png'%(path,name.split(".")[0].split("-")[0]))
    
    return output
    


#    
    

        
    
    
    