import importlib
import logging
import os
import shutil
import sys
from skimage import io

import h5py
from PIL import Image
import numpy as np
import scipy.sparse as sparse
import torch
import matplotlib.pyplot as plt
import uuid
# from sklearn.decomposition import PCA
import warnings
import pylab
import cv2
########################################################################################################################
plt.ioff()
plt.switch_backend('agg')

########################################################################################################################
def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]



def save_tiff_image(args, image_tensor, image_path):
    image = (image_tensor.cpu().detach().numpy()*args.normalize_factor) #.astype(np.uint16)
    # print('image max ',np.max(image),' image min ',np.min(image))
    '''
    image_tensor = torch.from_numpy(image_numpy)
    image_new_numpy = image_tensor.permute(2, 0, 1).numpy()
    # image_new_numpy = image_numpy
    if '_A' in image_path:
        image_new_numpy = image_new_numpy[...,np.newaxis] 
    '''
    save_tiff_path = image_path.replace('.png','.tif')
    io.imsave(save_tiff_path,image)

def save_feature_tiff_image(image_tensor, image_path):
    image = image_tensor.cpu().detach().numpy()*255 #).astype(np.uint8)
    # image = np.clip(image, 0, 65535).astype('uint16')
    image = np.clip(image, 0, 255).astype('uint8')
    # print('image max ',np.max(image),' image min ',np.min(image))
    save_tiff_path = image_path.replace('.png','.tif')
    io.imsave(save_tiff_path,image)



def FFDrealign4(input):
    # batch channel time height width
    realign_input = torch.cuda.FloatTensor(input.shape[0], input.shape[1]*4, input.shape[2], int(input.shape[3]/2), int(input.shape[4]/2))
    # print('realign_input -----> ',realign_input.shape)
    # print('input -----> ',input.shape)
    realign_input[:,0,:,:,:] = input[:,:,:,0::2, 0::2]
    realign_input[:,1,:,:,:] = input[:,:,:,0::2, 1::2]
    realign_input[:,2,:,:,:] = input[:,:,:,1::2, 0::2]
    realign_input[:,3,:,:,:] = input[:,:,:,1::2, 1::2]
    return realign_input

def inv_FFDrealign4(input):
    # batch channel time height width
    realign_input = torch.cuda.FloatTensor(input.shape[0], int(input.shape[1]/4), input.shape[2], int(input.shape[3]*2), int(input.shape[4]*2))

    realign_input[:,:,:,0::2, 0::2] = input[:,0,:,:,:]
    realign_input[:,:,:,0::2, 1::2] = input[:,1,:,:,:]
    realign_input[:,:,:,1::2, 0::2] = input[:,2,:,:,:]
    realign_input[:,:,:,1::2, 1::2] = input[:,3,:,:,:]
    return realign_input