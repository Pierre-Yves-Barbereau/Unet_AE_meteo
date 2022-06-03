#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:43:55 2022

@author: barbereaup
"""

import math
import random
import torch.nn as nn
import torch
import numpy as np
from torch.optim import Adam
import time
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Unet_AE_meteo(nn.Module):
    def __init__(self):
        super(Unet_AE_meteo,self).__init__()
        self.input_channels=1
        self.output_channels=1
        #V1
        self.bnout = nn.BatchNorm2d(self.output_channels)
        self.bn28 = nn.BatchNorm2d(28)
        self.bn56 = nn.BatchNorm2d(56)
        self.bn112 = nn.BatchNorm2d(112)
        self.bn224 = nn.BatchNorm2d(224)
        self.pad = nn.ReplicationPad2d(1)
        self.convin_56 = nn.Conv2d(in_channels=self.input_channels, out_channels=56, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv56_56 = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1,padding=0,bias=None)
        self.convin_56bis = nn.Conv2d(in_channels=self.input_channels, out_channels=56, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv14_14 = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv28_28 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv56_56bis = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv56_112 = nn.Conv2d(in_channels=56, out_channels=112, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv112_224 = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv224_448 = nn.Conv2d(in_channels=224, out_channels=448, kernel_size=3, stride=1,padding=0,bias=None)
        
        self.conv224_224 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1,padding=0,bias=None)
        
        self.conv112_112 = nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3, stride=1,padding=0,bias=None)
        
        self.conv56_3 = nn.Conv2d(in_channels=56, out_channels=3, kernel_size=3, stride=1,padding=0,bias=None)
        self.lrelu = nn.LeakyReLU(0.1)
        self.maxpool2 = nn.MaxPool2d((2,2),2)
        
         #V2
        self.conv448_896 = nn.Conv2d(in_channels=448, out_channels=896, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv896_1792 = nn.Conv2d(in_channels=896, out_channels=1792, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv1792_896 = nn.Conv2d(in_channels=1792, out_channels=896, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv1792_896bis = nn.Conv2d(in_channels=1792, out_channels=896, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv896_448 = nn.Conv2d(in_channels=896, out_channels=448, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv896_448bis = nn.Conv2d(in_channels=896, out_channels=448, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv448_224 = nn.Conv2d(in_channels=448, out_channels=224, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv448_224bis = nn.Conv2d(in_channels=448, out_channels=224, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv224_112 = nn.Conv2d(in_channels=224, out_channels=112, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv224_112bis = nn.Conv2d(in_channels=224, out_channels=112, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv112_56 = nn.Conv2d(in_channels=112, out_channels=56, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv112_56bis = nn.Conv2d(in_channels=112, out_channels=56, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv112_56ter = nn.Conv2d(in_channels=112, out_channels=56, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv56_28 = nn.Conv2d(in_channels=56, out_channels=28, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv56_28bis = nn.Conv2d(in_channels=56, out_channels=28, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv28_14 = nn.Conv2d(in_channels=28, out_channels=14, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv28_14bis = nn.Conv2d(in_channels=28, out_channels=14, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv14_14 = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv14_14bis = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv14_14ter = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv14_7 = nn.Conv2d(in_channels=14, out_channels=7, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv14_7bis = nn.Conv2d(in_channels=14, out_channels=7, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv56_out = nn.Conv2d(in_channels=56, out_channels=self.output_channels, kernel_size=3, stride=1,padding=0,bias=None)
#        self.conv1792_3584 = nn.Conv2d(in_channels=1792, out_channels=3584, kernel_size=3, stride=1,padding=0,bias=None)
        
      
        
#        self.conv3584_1792 = nn.Conv2d(in_channels=3584, out_channels=1792, kernel_size=3, stride=1,padding=0,bias=None)
#        self.conv1792_896 = nn.Conv2d(in_channels=1792, out_channels=896, kernel_size=3, stride=1,padding=0,bias=None)
#        self.conv896_448=nn.Conv2d(in_channels=896, out_channels=448, kernel_size=3, stride=1,padding=0,bias=None)
        
#        self.conv896_448=nn.Conv2d(in_channels=896, out_channels=448, kernel_size=3, stride=1,padding=0,bias=None)
        

        
       
    def forward(self, x):

        d0 = self.lrelu(self.conv56_28(self.pad(self.convin_56(self.pad(nn.functional.interpolate(x,size=(214,267),mode='bicubic'))))))
#        d0 = nn.functional.pad(d0, (1,1,0,0), mode='replicate')
#        dm1 = self.lrelu(self.conv28_14(self.pad(self.conv28_28(self.pad(nn.functional.interpolate(d0,size=(107,133),mode='bicubic'))))))
#
#        dm2 = self.lrelu(self.conv14_7(self.pad(self.conv14_14(self.pad(nn.functional.interpolate(dm1,size=(214,267),mode='bicubic'))))))

        d1 = self.lrelu(self.bn56(self.conv56_56(self.pad(self.bn56(self.convin_56bis(self.pad(x)))))))
        d1 = nn.functional.pad(d1, (0,1,0,0), mode='replicate')

        d2 = nn.functional.pad(d1, (0,1,0,1), mode='replicate')
        d2 = self.lrelu(self.bn112(self.conv56_112(self.pad(self.maxpool2(d2)))))
        d2 = d2[:,:,:,:27]


        d3 = self.lrelu(self.bn224(self.conv112_224(self.pad(self.maxpool2(d2)))))

#        d4 = self.lrelu(self.conv224_448(self.pad(self.maxpool2(d3))))
#        print("d4.shape = ",d4.shape)
#        d5 = self.lrelu(self.conv448_896(self.pad(self.maxpool2(d4))))
#        d6 = self.lrelu(self.conv896_1792(self.pad(self.maxpool2(d5))))
#        
#        x = nn.functional.interpolate(d6,scale_factor=2,mode='bicubic')
#        x = self.lrelu(self.conv1792_896(self.pad(x)))
#        x = torch.cat((d5,x),dim=1)
#        x = self.lrelu(self.conv1792_896bis(self.pad(x)))
#        
#        x = nn.functional.interpolate(x,scale_factor=2,mode='bicubic')
#        x = self.lrelu(self.conv896_448(self.pad(x)))
#        x = torch.cat((d4,x),dim=1)
#        x = self.lrelu(self.conv896_448bis(self.pad(x)))
        
#        x = nn.functional.interpolate(d4,scale_factor=2,mode='bicubic')
#        x = self.lrelu(self.conv448_224(self.pad(x)))
#        x = nn.functional.pad(x, (1,0,0,0), mode='replicate')
        
#        x = torch.cat((d3,x),dim=1)
#        x = self.lrelu(self.conv448_224bis(self.pad(x)))
#        
        x = nn.functional.interpolate(d3,scale_factor=2,mode='bicubic')
        x = self.lrelu(self.bn112(self.conv224_112(self.pad(x))))
        
        x = nn.functional.pad(x, (1,0,1,0), mode='replicate')
        x = x[:,:,:,:27]
#        print("x.shape = ",x.shape)
        x = torch.cat((d2,x),dim=1)
        x = self.lrelu(self.bn112(self.conv224_112bis(self.pad(x))))

        x = nn.functional.interpolate(x,scale_factor=2,mode='bicubic')
        x = self.lrelu(self.bn56(self.conv112_56(self.pad(x))))
#        print("x.shape = ",x.shape)
        x = torch.cat((d1,x),dim=1)
        x = self.lrelu(self.bn56(self.conv112_56bis(self.pad(x))))
        
        x = nn.functional.interpolate(x,size=(214,267),mode='bicubic')
        x = self.lrelu(self.bn28(self.conv56_28(self.pad(x))))
        
        x = torch.cat((d0,x),dim=1)
        x = self.lrelu(self.bnout(self.conv56_out(self.pad(self.lrelu(self.bn56(self.conv56_56bis(self.pad(x))))))))
        
        
        return x
    


    
    def fit(self, train_input_batch,train_output_batch,test_input_batch,test_output_batch, EPOCHS,batch_size=16,mode='normal'):
        startTime = time.time()
        lr = batch_size*1e-4
        opt = Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
        train_loss_list = []
        test_loss_list = []
        MAE_test_loss_list = []
        
        #Training loop
        for e in range(0, EPOCHS):
            # loop over the training set
            n = 0
            nmax=train_input_batch.shape[0]/batch_size
            
            perm_train = torch.randperm(train_output_batch.shape[0])
            perm_test = torch.randperm(test_output_batch.shape[0])
            
            train_input_batch = train_input_batch[perm_train]
            train_output_batch = train_output_batch[perm_train]
            test_input_batch = test_input_batch[perm_test]
            test_output_batch = test_output_batch[perm_test]
            
#            if e==30:
#                opt=SGD(self.parameters(), lr=1e-4)
#                scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
            
            if e%5==0:
                scheduler.step()
            
            #bath size split
            for (x,y) in zip(train_input_batch.split(batch_size),train_output_batch.split(batch_size)):

                #training
                
                n=n+1
                
                pred = self.forward(x.cuda())
                print("pred.shape = ",pred.shape)
                print("y.shape = ",y.shape)
                
                #ùetrics
               
                MAE_train = torch.abs((y.cuda()-pred)).mean()
                #
                ssim_train = -self.ssim(pred,y.cuda())
                
                mge_train = self.MGE_loss(pred,y.cuda())
                
                lap = self.Laplacian(pred,y)
                
                loss = 0.1*mge_train + 0.1*ssim_train + 0.1*lap + MAE_train + 0.1
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss_list.append(loss.item())
                
                
                
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                
                
#                
#                randint=random.randrange(0,len(test_output_batch.split(batch_size)))    #rend test sample permutation
#                with torch.no_grad():
#                    pred_test=self.forward(test_input_batch.split(batch_size)[randint].cuda())
#                MSE_test=((test_output_batch.split(batch_size)[randint].cuda()-pred_test)**2).mean()
#                train_loss_list.append(MSE_train.item())
#                ssim_test=-ssim(self.forward(test_input_batch.split(batch_size)[randint].cuda()),test_output_batch.split(batch_size)[randint].cuda())
#
#                mge_test=MGE_loss(self.forward(test_input_batch.split(batch_size)[randint].cuda()),test_output_batch.split(batch_size)[randint].cuda(),batch_size=batch_size)
#                #test_loss_list.append(test_loss.item())
#                test_loss=0.1*mge_test+0.1*ssim_test+MSE_test+0.1
#                test_loss_list.append(MSE_test.item())
                
                
                
#                if loss.item()<test_loss.item():
#                    overfitting_counter+=1
#                else:
#                    overfitting_counter=0
#                if overfitting_counter>30:
#                    print("overfitting_breaked")
#                    break
                
                
             
                
                ##test loss##
                randint=random.randrange(0,len(test_output_batch.split(batch_size)))  # select random indice of batch
                with torch.no_grad():
                    pred_test = self.forward(test_input_batch.split(batch_size)[randint].cuda())
                test_target = test_output_batch.split(batch_size)[randint].cuda()
                MAE_test = torch.abs((test_target-pred_test)).mean()
                ssim_test = -self.ssim(pred_test,test_target)
                mge_test = self.MGE_loss(pred_test,test_target)
                lap_test = self.Laplacian(pred_test,test_target)
                test_loss = 0.1*mge_test + 0.1 * ssim_test + MAE_test + 0.1
                test_loss_list.append(test_loss.item())
                MAE_test_loss_list.append(MAE_test.item())
                
                endTime = time.time()
                Time=endTime-startTime
                print("\n Training : time elapsed = ",Time," \n epoch = ",e,"/",EPOCHS, " nbatch = ",n,"/",nmax) 
                print("loss train =",loss.item())
                print("MGE train= ",mge_train.item())
                print("MSE train= ",MAE_train.item())
                print("loss test  =",test_loss.item())
                print("MGE test= ",mge_test.item())
                print("MSE test= ",MAE_test.item())
               
               
        return train_loss_list,test_loss_list,MAE_test_loss_list
    
    def predict(self,input_batch,output_batch,rescale):
        
        pred_list = []
        visu_list=[]
        n = 0
        nmax = input_batch.shape[0]
        for im_input,im_output in zip(input_batch,output_batch):
            n = n+1
            print(" predict n = ",n,"/",nmax)
            im_input = im_input.unsqueeze(0)
            pred = self.forward(im_input.cuda()).cpu()[0]
          
            #im_output_visu = torch.cat((nn.functional.interpolate(im_input,size=(214,267),mode='bicubic')[0],pred,im_output),dim=2)
            pred = (pred.permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            interp=(nn.functional.interpolate(im_input,size=(214,267),mode='bicubic')[0].permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            im_input=(im_input[0].permute(1,2,0).detach().numpy()*rescale)[:,:,0]            
            
            pred_list.append(pred)
            im_output=(im_output.permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            visu_list.append([im_input,interp,pred,im_output])
            #saving image
            #im_out.save(f"%s/%s-predicted.jpeg" %(dir_name,name))
        return np.asarray(pred_list),np.asarray(visu_list)
    def gaussian(self,window_size, sigma):
        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.
    
        Length of list = window_size
        """    
        gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        
    def create_window(self,window_size, channel=3):
    
        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = self.gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
         
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
    
        return window
    
                
    def ssim(self,img1, img2, window_size=11, val_range=255, window=None, size_average=True, full=False):
    
        L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),
    
        pad = window_size // 2
        
        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()
    
        # if window is not provided, init one
        if window is None: 
            real_size = min(window_size, height, width) # window should be atleast 11x11 
            window = self.create_window(real_size, channel=channels).to(img1.device)
        
        # calculating the mu parameter (locally) for both images using a gaussian filter 
        # calculates the luminosity params
        mu1 = nn.functional.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = nn.functional.conv2d(img2, window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2 
        mu12 = mu1 * mu2
    
        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component 
        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 =  nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12
    
        # Some constants for stability 
        C1 = (0.01*L ) ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = (0.03*L ) ** 2 
    
        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)
    
        numerator1 = 2 * mu12 + C1  
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1 
        denominator2 = sigma1_sq + sigma2_sq + C2
    
        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
    
        if size_average:
            ret = ssim_score.mean() 
        else: 
            ret = ssim_score.mean(1).mean(1).mean(1)
        
        if full:
            return ret, contrast_metric
        
        return ret      
    
    def MGE_loss(self,pred,target):
        output_channels=self.output_channels
    #    x_filter = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).unsqueeze(0).repeat(batch_size,3,1,1).float().cuda()
    #    y_filter = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).unsqueeze(0).repeat(1,3,1,1).float().cuda()
    #    x_filter = torch.Tensor([[[[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]]]]).cuda()
    #    y_filter = torch.Tensor([[[[-1,-2,-1]], [[0,0,0]], [[1,2,1]]]]).cuda()
        x_filter = torch.Tensor([[[[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]]]]).repeat(1,output_channels,1,1).cuda()
        y_filter = torch.Tensor([[[[-1,-2,-1], [0,0,0], [1,2,1]]]]).repeat(1,output_channels,1,1).cuda()
        pad = nn.ReplicationPad2d(1)
        pred = pad(pred)
        target = pad(target)
        pred_gx = nn.functional.conv2d(pred,x_filter,stride=1, padding=0).cpu().detach().numpy()
        pred_gy = nn.functional.conv2d(pred, y_filter,stride=1, padding=0).cpu().detach().numpy()
        target_gx = nn.functional.conv2d(target, x_filter, padding=0).cpu().detach().numpy()
        target_gy = nn.functional.conv2d(target, y_filter, padding=0).cpu().detach().numpy()
        g_pred = np.sqrt(pred_gx**2 + pred_gy**2)
        g_target = np.sqrt(target_gx**2 + target_gy**2)    
        return ((g_pred - g_target)**2).mean()

    def Laplacian(self,pred,target):
        output_channels = self.output_channels
        filters = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]]).repeat(1,output_channels,1,1).float().cuda()
        pad = nn.ReplicationPad2d(1)
        pred = pad(pred)
        target = pad(target)
        lpred = nn.functional.conv2d(pred,filters,stride=1,padding=0).cpu().detach().numpy()
        ltarget = nn.functional.conv2d(pred,filters,stride=1,padding=0).cpu().detach().numpy()
        return np.abs(lpred - ltarget).mean()
    
    
    
    
    
    
    
    
    
class Unet_AE_meteo2(nn.Module):
    def __init__(self):
        super(Unet_AE_meteo2,self).__init__()
        self.input_channels=1
        self.output_channels=1
        #V1
        self.bnout = nn.BatchNorm2d(self.output_channels)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn1024 = nn.BatchNorm2d(1024)
        self.pad = nn.ReplicationPad2d(1)
        self.convin_128 = nn.Conv2d(in_channels=self.input_channels, out_channels=128, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv128_128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=0,bias=None)
        self.convin_256 = nn.Conv2d(in_channels=self.input_channels, out_channels=256, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv256_256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv256_256bis = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv256_512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv512_1024 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv1024_512 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv1024_512bis = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv512_256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv512_256bis = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv256_128 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv256_128bis = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1,padding=0,bias=None)
        self.conv128_out = nn.Conv2d(in_channels=128, out_channels=self.output_channels, kernel_size=3, stride=1,padding=0,bias=None)
        self.lrelu = nn.LeakyReLU(0.1)
        self.maxpool2 = nn.MaxPool2d((2,2),2)
        
#        self.conv1792_3584 = nn.Conv2d(in_channels=1792, out_channels=3584, kernel_size=3, stride=1,padding=0,bias=None)
        
      
        
#        self.conv3584_1792 = nn.Conv2d(in_channels=3584, out_channels=1792, kernel_size=3, stride=1,padding=0,bias=None)
#        self.conv1792_896 = nn.Conv2d(in_channels=1792, out_channels=896, kernel_size=3, stride=1,padding=0,bias=None)
#        self.conv896_448=nn.Conv2d(in_channels=896, out_channels=448, kernel_size=3, stride=1,padding=0,bias=None)
        
#        self.conv896_448=nn.Conv2d(in_channels=896, out_channels=448, kernel_size=3, stride=1,padding=0,bias=None)
        

        
       
    def forward(self, x):

        d0 = self.bn128(self.lrelu(self.conv128_128(self.pad(self.bn128(self.lrelu(self.convin_128(self.pad(nn.functional.interpolate(x,size=(214,267),mode='bicubic')))))))))
#        d0 = nn.functional.pad(d0, (1,1,0,0), mode='replicate')
#        dm1 = self.lrelu(self.conv28_14(self.pad(self.conv28_28(self.pad(nn.functional.interpolate(d0,size=(107,133),mode='bicubic'))))))
#
#        dm2 = self.lrelu(self.conv14_7(self.pad(self.conv14_14(self.pad(nn.functional.interpolate(dm1,size=(214,267),mode='bicubic'))))))

        d1 = self.bn256(self.lrelu(self.conv256_256(self.pad(self.bn256(self.lrelu(self.convin_256(self.pad(x))))))))
        d1 = nn.functional.pad(d1, (0,1,0,0), mode='replicate')

        d2 = nn.functional.pad(d1, (0,1,0,1), mode='replicate')
        d2 = self.bn512(self.lrelu(self.conv256_512(self.pad(self.maxpool2(d2)))))
        d2 = d2[:,:,:,:27]


        d3 = self.bn1024(self.lrelu(self.conv512_1024(self.pad(self.maxpool2(d2)))))

#        d4 = self.lrelu(self.conv224_448(self.pad(self.maxpool2(d3))))
#        print("d4.shape = ",d4.shape)
#        d5 = self.lrelu(self.conv448_896(self.pad(self.maxpool2(d4))))
#        d6 = self.lrelu(self.conv896_1792(self.pad(self.maxpool2(d5))))
#        
#        x = nn.functional.interpolate(d6,scale_factor=2,mode='bicubic')
#        x = self.lrelu(self.conv1792_896(self.pad(x)))
#        x = torch.cat((d5,x),dim=1)
#        x = self.lrelu(self.conv1792_896bis(self.pad(x)))
#        
#        x = nn.functional.interpolate(x,scale_factor=2,mode='bicubic')
#        x = self.lrelu(self.conv896_448(self.pad(x)))
#        x = torch.cat((d4,x),dim=1)
#        x = self.lrelu(self.conv896_448bis(self.pad(x)))
        
#        x = nn.functional.interpolate(d4,scale_factor=2,mode='bicubic')
#        x = self.lrelu(self.conv448_224(self.pad(x)))
#        x = nn.functional.pad(x, (1,0,0,0), mode='replicate')
        
#        x = torch.cat((d3,x),dim=1)
#        x = self.lrelu(self.conv448_224bis(self.pad(x)))
#        
        x = nn.functional.interpolate(d3,scale_factor=2,mode='bicubic')
        x = self.bn512(self.lrelu(self.conv1024_512(self.pad(x))))
        
        x = nn.functional.pad(x, (1,0,1,0), mode='replicate')
        x = x[:,:,:,:27]
#        print("x.shape = ",x.shape)
        x = torch.cat((d2,x),dim=1)
        x = self.bn512(self.lrelu(self.conv1024_512bis(self.pad(x))))

        x = nn.functional.interpolate(x,scale_factor=2,mode='bicubic')
        x = self.bn256(self.lrelu(self.conv512_256(self.pad(x))))
#        print("x.shape = ",x.shape)
        x = torch.cat((d1,x),dim=1)
        x = self.bn256(self.lrelu(self.conv512_256bis(self.pad(x))))
        
        x = nn.functional.interpolate(x,size=(214,267),mode='bicubic')
        x = self.bn128(self.lrelu(self.conv256_128(self.pad(x))))
        x = torch.cat((d0,x),dim=1)
        x = self.bnout(self.lrelu(self.conv128_out(self.pad(self.bn128(self.lrelu(self.conv256_128bis(self.pad(x))))))))
        
        
        return x
    


    
    def fit(self, train_input_batch,train_output_batch,test_input_batch,test_output_batch, EPOCHS,batch_size=16,mode='normal'):
        startTime = time.time()
        lr = batch_size*1e-4
        opt = Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
        train_loss_list = []
        test_loss_list = []
        MAE_test_loss_list = []
        
        #Training loop
        for e in range(0, EPOCHS):
            # loop over the training set
            n = 0
            nmax=train_input_batch.shape[0]/batch_size
            
            perm_train = torch.randperm(train_output_batch.shape[0])
            perm_test = torch.randperm(test_output_batch.shape[0])
            
            train_input_batch = train_input_batch[perm_train]
            train_output_batch = train_output_batch[perm_train]
            test_input_batch = test_input_batch[perm_test]
            test_output_batch = test_output_batch[perm_test]
            
#            if e==30:
#                opt=SGD(self.parameters(), lr=1e-4)
#                scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
            
            if e%5==0:
                scheduler.step()
            
            #bath size split
            for (x,y) in zip(train_input_batch.split(batch_size),train_output_batch.split(batch_size)):

                #training
                
                n=n+1
                
                pred = self.forward(x.cuda())
                print("pred.shape = ",pred.shape)
                print("y.shape = ",y.shape)
                
                #ùetrics
               
                MAE_train = torch.abs((y.cuda()-pred)).mean()
                #
                ssim_train = -self.ssim(pred,y.cuda())
                
                mge_train = self.MGE_loss(pred,y.cuda())
                
                lap = self.Laplacian(pred,y)
                
                loss = 0.1*mge_train + 0.1*ssim_train + 0.1*lap + MAE_train + 0.1
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss_list.append(loss.item())
                
                
                
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                
                
#                
#                randint=random.randrange(0,len(test_output_batch.split(batch_size)))    #rend test sample permutation
#                with torch.no_grad():
#                    pred_test=self.forward(test_input_batch.split(batch_size)[randint].cuda())
#                MSE_test=((test_output_batch.split(batch_size)[randint].cuda()-pred_test)**2).mean()
#                train_loss_list.append(MSE_train.item())
#                ssim_test=-ssim(self.forward(test_input_batch.split(batch_size)[randint].cuda()),test_output_batch.split(batch_size)[randint].cuda())
#
#                mge_test=MGE_loss(self.forward(test_input_batch.split(batch_size)[randint].cuda()),test_output_batch.split(batch_size)[randint].cuda(),batch_size=batch_size)
#                #test_loss_list.append(test_loss.item())
#                test_loss=0.1*mge_test+0.1*ssim_test+MSE_test+0.1
#                test_loss_list.append(MSE_test.item())
                
                
                
#                if loss.item()<test_loss.item():
#                    overfitting_counter+=1
#                else:
#                    overfitting_counter=0
#                if overfitting_counter>30:
#                    print("overfitting_breaked")
#                    break
                
                
             
                
                ##test loss##
                randint=random.randrange(0,len(test_output_batch.split(batch_size)))  # select random indice of batch
                with torch.no_grad():
                    pred_test = self.forward(test_input_batch.split(batch_size)[randint].cuda())
                test_target = test_output_batch.split(batch_size)[randint].cuda()
                MAE_test = torch.abs((test_target-pred_test)).mean()
                ssim_test = -self.ssim(pred_test,test_target)
                mge_test = self.MGE_loss(pred_test,test_target)
                lap_test = self.Laplacian(pred_test,test_target)
                test_loss = 0.1*mge_test + 0.1*ssim_test + 0.1*lap_test + MAE_test + 0.1
                test_loss_list.append(test_loss.item())
                MAE_test_loss_list.append(MAE_test.item())
                
                endTime = time.time()
                Time=endTime-startTime
                print("\n Training : time elapsed = ",Time," \n epoch = ",e,"/",EPOCHS, " nbatch = ",n,"/",nmax) 
                print("loss train =",loss.item())
                print("MGE train= ",mge_train.item())
                print("MSE train= ",MAE_train.item())
                print("loss test  =",test_loss.item())
                print("MGE test= ",mge_test.item())
                print("MSE test= ",MAE_test.item())
               
               
        return train_loss_list,test_loss_list,MAE_test_loss_list
    
    def predict(self,input_batch,output_batch,rescale):
        
        pred_list = []
        visu_list=[]
        n = 0
        nmax = input_batch.shape[0]
        for im_input,im_output in zip(input_batch,output_batch):
            n = n+1
            print(" predict n = ",n,"/",nmax)
            im_input = im_input.unsqueeze(0)
            pred = self.forward(im_input.cuda()).cpu()[0]
          
            #im_output_visu = torch.cat((nn.functional.interpolate(im_input,size=(214,267),mode='bicubic')[0],pred,im_output),dim=2)
            pred = (pred.permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            interp=(nn.functional.interpolate(im_input,size=(214,267),mode='bicubic')[0].permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            im_input=(im_input[0].permute(1,2,0).detach().numpy()*rescale)[:,:,0]            
            
            pred_list.append(pred)
            im_output=(im_output.permute(1,2,0).detach().numpy()*rescale)[:,:,0]
            visu_list.append([im_input,interp,pred,im_output])
            #saving image
            #im_out.save(f"%s/%s-predicted.jpeg" %(dir_name,name))
        return np.asarray(pred_list),np.asarray(visu_list)
    def gaussian(self,window_size, sigma):
        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.
    
        Length of list = window_size
        """    
        gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        
    def create_window(self,window_size, channel=3):
    
        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = self.gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
         
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
    
        return window
    
                
    def ssim(self,img1, img2, window_size=11, val_range=255, window=None, size_average=True, full=False):
    
        L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),
    
        pad = window_size // 2
        
        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()
    
        # if window is not provided, init one
        if window is None: 
            real_size = min(window_size, height, width) # window should be atleast 11x11 
            window = self.create_window(real_size, channel=channels).to(img1.device)
        
        # calculating the mu parameter (locally) for both images using a gaussian filter 
        # calculates the luminosity params
        mu1 = nn.functional.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = nn.functional.conv2d(img2, window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2 
        mu12 = mu1 * mu2
    
        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component 
        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 =  nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12
    
        # Some constants for stability 
        C1 = (0.01*L ) ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = (0.03*L ) ** 2 
    
        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)
    
        numerator1 = 2 * mu12 + C1  
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1 
        denominator2 = sigma1_sq + sigma2_sq + C2
    
        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
    
        if size_average:
            ret = ssim_score.mean() 
        else: 
            ret = ssim_score.mean(1).mean(1).mean(1)
        
        if full:
            return ret, contrast_metric
        
        return ret      
    
    def MGE_loss(self,pred,target):
        output_channels=self.output_channels
    #    x_filter = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).unsqueeze(0).repeat(batch_size,3,1,1).float().cuda()
    #    y_filter = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).unsqueeze(0).repeat(1,3,1,1).float().cuda()
    #    x_filter = torch.Tensor([[[[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]]]]).cuda()
    #    y_filter = torch.Tensor([[[[-1,-2,-1]], [[0,0,0]], [[1,2,1]]]]).cuda()
        x_filter = torch.Tensor([[[[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]]]]).repeat(1,output_channels,1,1).cuda()
        y_filter = torch.Tensor([[[[-1,-2,-1], [0,0,0], [1,2,1]]]]).repeat(1,output_channels,1,1).cuda()
        pad = nn.ReplicationPad2d(1)
        pred = pad(pred)
        target = pad(target)
        pred_gx = nn.functional.conv2d(pred,x_filter,stride=1, padding=0).cpu().detach().numpy()
        pred_gy = nn.functional.conv2d(pred, y_filter,stride=1, padding=0).cpu().detach().numpy()
        target_gx = nn.functional.conv2d(target, x_filter, padding=0).cpu().detach().numpy()
        target_gy = nn.functional.conv2d(target, y_filter, padding=0).cpu().detach().numpy()
        g_pred = np.sqrt(pred_gx**2 + pred_gy**2)
        g_target = np.sqrt(target_gx**2 + target_gy**2)    
        return ((g_pred - g_target)**2).mean()

    def Laplacian(self,pred,target):
        output_channels = self.output_channels
        filters = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]]).repeat(1,output_channels,1,1).float().cuda()
        pad = nn.ReplicationPad2d(1)
        pred = pad(pred)
        target = pad(target)
        lpred = nn.functional.conv2d(pred,filters,stride=1,padding=0).cpu().detach().numpy()
        ltarget = nn.functional.conv2d(pred,filters,stride=1,padding=0).cpu().detach().numpy()
        return np.abs(lpred - ltarget).mean()
