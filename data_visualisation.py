#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:08:32 2022

@author: barbereaup
"""

import inspect
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import Models as rnn
from mpl_toolkits.axes_grid1 import ImageGrid
#data500,data1200,name_list_test=data_extraction.fulldataload(["temp850"])
def array_loader(path):
    name_list=os.listdir(path)
    array_list=[]
    for name in name_list:
        if name.split(".")[-1]=="npy":
            
            array_list.append(np.load(f"%s/%s"%(path,name),allow_pickle=True))
    return array_list


    
def visualisation(array,show=True):
    fig = plt.figure(figsize=(50, 50))
    grid = ImageGrid(fig, 142,  # similar to subplot(142)
                     nrows_ncols=(2, 2),
                     axes_pad=0.0,
#                     share_all=True,
                     cbar_mode="single",
                     cbar_location="right",
#                     label_mode = "all",
                     )
    extent= (-3, 4, -4, 3)
    for i, ax, im in zip(range(4),grid, [array[0], array[1], array[2], array[3]]):
    # Iterating over the grid returns the Axes.
        im0 = ax.imshow(im,extent=extent)
        grid.cbar_axes[i].colorbar(im0)

    for cax in grid.cbar_axes:
        cax.toggle_label(True)

    # This affects all axes as share_all = True.
#    grid.axes_llc.set_xticks([-2, 0, 2])
#    grid.axes_llc.set_yticks([-2, 0, 2])
    if show==True:
        plt.show()
    
def save_data(array_list,name_list_test):
    for array in array_list:
        fig = plt.figure(figsize=(50, 50))
        grid = ImageGrid(fig, 142,  # similar to subplot(142)
                         nrows_ncols=(2, 2),
                         axes_pad=0.0,
    #                     share_all=True,
                         cbar_mode="single",
                         cbar_location="right",
    #                     label_mode = "all",
                         )
        extent= (-3, 4, -4, 3)
        for i, ax, im in zip(range(4),grid, [array[0], array[1], array[2], array[3]]):
        # Iterating over the grid returns the Axes.
            im0 = ax.imshow(im,extent=extent)
            grid.cbar_axes[i].colorbar(im0)
    
        for cax in grid.cbar_axes:
            cax.toggle_label(True)
        