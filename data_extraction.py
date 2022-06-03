#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:35:24 2022

@author: barbereaup
"""


import numpy as np
import os


#name_list=os.listdir(path)
def dataload(path500,path1200):
    name_list=os.listdir(path500)
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
