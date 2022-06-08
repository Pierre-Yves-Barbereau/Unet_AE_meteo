#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:43:26 2022

@author: barbereaup
"""
EPOCHS = 30
batch_size = 32

import inspect
import os
#from torch.nn.init import orthogonal_
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import data_extraction
import Models as rnn
import data_visualisation as dv
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_channels = ["temp850"]
output_channel_indice = 0
data500,data1200,name_list_test=data_extraction.fulldataload(input_channels)
rescale=max(data500.max(),data1200.max())

print("data1200.shape = ",data1200.shape)
print("data500.shape = ",data500.shape)
data500 = torch.from_numpy(data500).float()/rescale
data1200 = torch.from_numpy(data1200).float()/rescale
train500 = data500[:int(len(data500)*0.8)]
test500 = data500[int(len(data500)*0.8)+1:]
train1200 = data1200[:int(len(data1200)*0.8)]
test1200 = data1200[int(len(data1200)*0.8)+1:]




#train500 = torch.from_numpy(np.load("../data/train_HR.npy",allow_pickle=True)).permute(0,3,1,2).float()/rescale
#train1200 = torch.from_numpy(np.load("../data/train_LR.npy",allow_pickle=True)).permute(0,3,1,2).float()/rescale
#name_list_test = np.load("../data/name_list.npy")
#test500 = torch.from_numpy(np.load("../data/test_HR.npy",allow_pickle=True)).permute(0,3,1,2).float()/rescale
#test1200 = torch.from_numpy(np.load("../data/test_LR.npy",allow_pickle=True)).permute(0,3,1,2).float()/rescale
#
#
#
#train500 = train500[:,:,148:362,122:389].mean(axis=1).unsqueeze(1)
#train1200 = train1200[:,:,107:149,101:154].mean(axis=1).unsqueeze(1)
#print("train_HR.shape = ",train500.shape)
#print("train_LR.shape = ",train1200.shape)
#test500 = test500[:,:,148:362,122:389].mean(axis=1).unsqueeze(1)
#test1200 = test1200[:,:,107:149,101:154].mean(axis=1).unsqueeze(1)


# initialize the model
print("[INFO] initializing the model...")
model = rnn.Unet_AE_meteo()
#orthogonal_(model.weight.data)
if torch.cuda.is_available():
    model.cuda()
print(model)
#summary(model, (3, 256, 256))

#def Orthogonal_Init(model, alsoLinear=False):
#    classname=model.__class__.__name__
#    if classname.find('Conv')!=-1:
#        orthogonal_(model.weight.data, gain=np.sqrt(2)) #gain to account for subsequ. ReLU
#    elif classname.find('Linear')!=-1 and alsoLinear :
#        orthogonal_(model.weight.data)
#
#model.apply(Orthogonal_Init)


# measure how long training is going to take
print("[INFO] training the network...")
StartTime = time.time()
train_loss_list,test_loss_list,MAE_test_loss_list=model.fit(train1200,train500,test1200,test500, EPOCHS = EPOCHS,batch_size=batch_size)
EndTime = time.time()


    
predict_test,predict_test_visu = model.predict(test1200,test500,rescale=rescale)


training_time=EndTime-StartTime
print("Training time = ",training_time)
x_axis = range(len(test_loss_list))
plt.plot(x_axis,train_loss_list,label="train")
plt.plot(x_axis,test_loss_list,label="test")
plt.legend()
plt.title('Test loss / epoch')
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()

x_axis = range(len(test_loss_list[-100:-1]))
plt.plot(x_axis,train_loss_list[-100:-1],label="train")
plt.plot(x_axis,test_loss_list[-100:-1],label="test")
plt.legend()
plt.title('Test loss / epoch')
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()

loss_test_mean_100 = np.mean(test_loss_list[-100:-1])
MAE_test_mean_100 = np.mean(MAE_test_loss_list[-100:-1])
print("loss_test_mean_100 = ",MAE_test_mean_100)


dv.visualisation(predict_test_visu[0])
save = input("Save results ?")

if save == 'y':

    cl=inspect.getsource(rnn.Unet_AE_meteo)
    #Make directory to store predicted images
    date = str(datetime.now()).split(".")[0].split(" ")[0]+"_"+str(datetime.now()).split(".")[0].split(" ")[1]
    dir_name=f"meteo_predict_%s" %date
    os.makedirs(dir_name)
    
    file = open(f"%s/resume.txt" %(dir_name),"w")
    file.write(f"%s \n input_channels = %s \n output_channels = %s \n loss = 0.1*mge + 0.1*ssim + 0.1lap +  MAE + 0.1 \nEPOCHS = %s \n Batch_size = %s \n loss_mean_last_100_it = %s \n MSE_mean_last_100_it = %s \n training time = %s  \n %s"%(dir_name,input_channels,input_channels[output_channel_indice],EPOCHS,batch_size,loss_test_mean_100,MAE_test_mean_100,training_time,cl))
    file.close()
    
    #saving loss
    plt.savefig(f"%s/loss.png" %(dir_name))
    #List the name of test images
    name_list_test = name_list_test[int(len(name_list_test)*0.8)+1:]
    for array,name in zip(predict_test_visu,name_list_test):
        dv.visualisation(array,show=False)
        plt.savefig(f"%s/%s_predicted.png" %(dir_name,name))
        plt.close()
    print("images saved in directory : ",dir_name)
    

    
