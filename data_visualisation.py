#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:08:32 2022

@author: barbereaup
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
#data500,data1200,name_list_test=data_extraction.fulldataload(["temp850"])
def array_loader(path):
    name_list=os.listdir(path)
    array_list=[]
    for name in name_list:
        if name.split(".")[-1]=="npy":
            
            array_list.append(np.load(f"%s/%s"%(path,name),allow_pickle=True))
    return array_list


    
def visualisation(array,bar=True,show=True):
    fig = plt.figure(figsize=(30, 30))
    if bar:
        grid = ImageGrid(fig, 142,  # similar to subplot(142)
                         nrows_ncols=(2, 2),
                         axes_pad=0.3,
    #                     share_all=True,
                         cbar_mode="single",
                         cbar_location="right",
    #                     label_mode = "all",
                         )
    else:
        grid = ImageGrid(fig, 142,  # similar to subplot(142)
                         nrows_ncols=(2, 2),
                         axes_pad=0.5,
    #                     share_all=True,
#                         cbar_mode="single",
#                         cbar_location="right",
    #                     label_mode = "all",
                         )
    extent= (-3, 4, -4, 3)
    title=["Original","Bicubic interpolation","Prediction","Target"]
    for i, ax, im in zip(range(4),grid, [array[0], array[1], array[2], array[3]]):
    # Iterating over the grid returns the Axes.
        im0 = ax.imshow(im,extent=extent)
        ax.set_title(title[i])
        if bar:
            grid.cbar_axes[i].colorbar(im0)
    if bar:
        for cax in grid.cbar_axes:
            cax.toggle_label(True)

    # This affects all axes as share_all = True.
#    grid.axes_llc.set_xticks([-2, 0, 2])
#    grid.axes_llc.set_yticks([-2, 0, 2])
#    plt.tight_layout()
    if show:
        plt.show()
        plt.close()
    return fig
    
            
def visualisation2(array,bar=True,show=True):
    fig = plt.figure(figsize=(30, 30))
    if bar:
        grid = ImageGrid(fig, 142,  # similar to subplot(142)
                         nrows_ncols=(2, 2),
                         axes_pad=0.5,
    #                     share_all=True,
                         cbar_mode="each",
                         cbar_location="right",
    #                     label_mode = "all",
                         )
    else:
        grid = ImageGrid(fig, 142,  # similar to subplot(142)
                         nrows_ncols=(2, 2),
                         axes_pad=0.5,
    #                     share_all=True,
#                         cbar_mode="single",
#                         cbar_location="right",
    #                     label_mode = "all",
                         )
    extent= (-8,8,-8,8)
    title=["original","interpolation bicubique","prediction","target"]
    for i, ax, im in zip(range(4),grid, [array[0], array[1], array[2], array[3]]):
    # Iterating over the grid returns the Axes.
        im0 = ax.imshow(im,extent=extent)
        ax.set_title(title[i])
        if bar:
            grid.cbar_axes[i].colorbar(im0)
    if bar:
        for cax in grid.cbar_axes:
            cax.toggle_label(True)

    # This affects all axes as share_all = True.
#    grid.axes_llc.set_xticks([-2, 0, 2])
#    grid.axes_llc.set_yticks([-2, 0, 2])
#    plt.tight_layout()
    if show:
        plt.show()
        plt.close()
        
def concat(array):
    output=[]
    cat1=np.concatenate([array[0],array[1]],axis=1)
#    print("cat1.shape = ",cat1.shape)
    cat2=np.concatenate([array[2],array[3]],axis=1)
#    print("cat2.shape = ",cat2.shape)
    output=np.concatenate([cat1,cat2],axis=0)
    return output

