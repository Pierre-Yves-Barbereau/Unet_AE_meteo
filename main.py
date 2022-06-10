#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:43:26 2022

@author: barbereaup
"""
import shutil
import inspect
import os
from torch.optim import Adam
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


EPOCHS = 30
batch_size = 8
lr = batch_size*1e-4
gamma=0.8
sheduler_period = 5




input_channels = ["t2m"]
output_channel_indice = 0
data500,data1200,name_list=data_extraction.fulldataload(input_channels)
rescale=max(data500.max(),data1200.max())

train1200,test1200,train500,test500,name_list_test=data_extraction.train_test_split(data1200,data500,name_list)
train1200 = torch.from_numpy(train1200).float()/rescale
test1200 = torch.from_numpy(test1200).float()/rescale
train500 = torch.from_numpy(train500).float()/rescale
test500 = torch.from_numpy(test500).float()/rescale


                    ###########Initialisation##########
print("[INFO] initializing the model...")
model = rnn.Unet_AE_meteo()
#orthogonal_(model.weight.data)
if torch.cuda.is_available():
    model.cuda()
print(model)



                        ############Training##########
                        
opt = Adam(model.parameters(), lr=lr)                       
print("[INFO] training the network...")
StartTime = time.time()
train_loss_list,test_loss_list=model.fit(train1200,
                                         train500,
                                         test1200,
                                         test500,
                                         EPOCHS = EPOCHS,
                                         opt = opt,
                                         gamma=gamma,
                                         sheduler_period=sheduler_period,
                                         batch_size=batch_size,
                                         )

EndTime = time.time()
training_time=EndTime-StartTime
print("Training time = ",training_time)


                    ############Predict##########
                    
                    
predict_test,predict_test_visu = model.predict(test1200,test500,rescale=rescale)

##Pre-visualisation

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
#MAE_test_mean_100 = np.mean(MAE_test_loss_list[-100:-1])
print("loss_test_mean_100 = ",loss_test_mean_100)
dv.visualisation(predict_test_visu[0])



                ##########Saving#########
                
                
                
save = input("Save results ?")
#save = 'y'
if save == 'y':
    
    
    cl = inspect.getsource(rnn.Unet_AE_meteo)
    #Make saving directory
    date = str(datetime.now()).split(".")[0].split(" ")[0]+"_"+str(datetime.now()).split(".")[0].split(" ")[1]
    dir_name = f"meteo_predict_%s" %date
    os.makedirs(dir_name)
    
    #Saving Hyper-parameters
    file = open(f"%s/resume.txt" %(dir_name),"w")
    file.write(f"%s \n input_channels = %s \n output_channels = %s \n loss = MSE  \nEPOCHS = %s \n Batch_size = %s \n MSE_mean_last_100_it = %s \n training time = %s  \n %s"%(dir_name,input_channels,input_channels[output_channel_indice],EPOCHS,batch_size,loss_test_mean_100,training_time,cl))
    file.close()
    
    #Saving script
    shutil.copy2('main.py', f"%s/main.py" %(dir_name))
    shutil.copy2('Models.py', f"%s/Models.py" %(dir_name))
    shutil.copy2('data_extraction.py', f"%s/data_extraction.py" %(dir_name))
    shutil.copy2('data_visualisation.py', f"%s/data_visualisation.py" %(dir_name))
    
    #Saving loss
    np.save("test_loss.npy",test_loss_list)
    np.save("train_loss.npy",train_loss_list)
    
    #Saving loss plot
    x_axis = range(len(test_loss_list))
    fig_loss = plt.figure(figsize=(10,10))
    sp = fig_loss.add_subplot(111)
    sp.plot(x_axis,train_loss_list,label="train")
    sp.plot(x_axis,test_loss_list,label="test")
    sp.legend()
    sp.set_title("Loss")
    sp.set_xscale("log")
    sp.set_yscale("log")
    fig_loss.show()
    fig_loss.savefig(f"%s/loss.png" %(dir_name),bbox_inches='tight')
    
    #Saving images prediction
    imax=predict_test_visu.shape[0]
    for array,name,i in zip(predict_test_visu,name_list_test,range(imax)):
        print(i,"/",imax)
        t1 = time.time()
        fig_visu = dv.visualisation(array,show=False)
        t2 = time.time()
        fig_visu.savefig(f"%s/%s_predicted.png" %(dir_name,name),dpi=300,bbox_inches='tight')
        t3 = time.time()
        plt.close()
        print("visu time = ",t2-t1)
        print("save time = ",t3-t2)
    print("images saved in directory : ",dir_name)
    

    
