import numpy as np
import argparse
import os
import tifffile as tiff
import time
import datetime
import random
from skimage import io
import logging
import math
from torch.utils.data import Dataset
import torch


#########################################################################
#########################################################################
class trainset_deepsemi(Dataset):
    def __init__(self, name_list, img_list, coordinate_list, para_dict):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.img_list = img_list
        self.para_dict = para_dict

    def __getitem__(self, index):
        #fn = self.images[index]
        patch_name = self.name_list[index]
        single_coordinate = self.coordinate_list[patch_name]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        img_name = single_coordinate['name']
        noise_img = self.img_list[img_name]
        noise_patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = random_transform(noise_patch)
        ######################################################
        cut_off = int((self.para_dict['in_c']-1)/2)
        input_patch = realign_patch(noise_patch, self.para_dict['in_c_s'])
        real_A = torch.from_numpy(input_patch).float()
        real_A = real_A.cuda()

        GT_patch = noise_patch[cut_off:-cut_off,:,:]
        real_B = torch.from_numpy(np.expand_dims(GT_patch, 0)).float()
        real_B = real_B.cuda()

        input_patch_t1 = realign_patch_t1(noise_patch, self.para_dict['in_c_t'])
        real_B_t1 = torch.from_numpy(input_patch_t1).float()
        real_B_t1 = real_B_t1.cuda()

        input_patch_t2 = realign_patch_t2(noise_patch, self.para_dict['in_c_t'])
        real_B_t2 = torch.from_numpy(input_patch_t2).float()
        real_B_t2 = real_B_t2.cuda()

        #target = self.target[index]
        # print(' real_A ---> ',real_A.shape,' real_B ---> ',real_B.shape,' real_B_t1 ---> ',real_B_t1.shape,' real_B_t2 ---> ',real_B_t2.shape)
        return real_A, real_B, real_B_t1, real_B_t2, patch_name

    def __len__(self):
        return len(self.name_list)


def realign_patch(patch, in_nc):
    re_patch = np.zeros((in_nc, patch.shape[0]-in_nc+1, patch.shape[1], patch.shape[2]))
    for i in range(0, patch.shape[0]-in_nc+1):
        re_patch[:,i,:,:] = patch[i:i+in_nc,:,:]
    return re_patch

def realign_patch_t1(patch, in_nc):
    re_patch = np.zeros((in_nc, patch.shape[0]-in_nc*2+2, patch.shape[1], patch.shape[2]))
    for i in range(0, patch.shape[0]-in_nc*2+2):
        re_patch[:,i,:,:] = patch[i:i+in_nc,:,:]
    return re_patch

def realign_patch_t2(patch, in_nc):
    re_patch = np.zeros((in_nc, patch.shape[0]-in_nc*2+2, patch.shape[1], patch.shape[2]))
    for i in range(in_nc-1, patch.shape[0]-in_nc+1):
        re_patch[:,i-in_nc+1,:,:] = patch[i:i+in_nc,:,:]
    return re_patch

def random_transform(noise_patch):
    flip_num = random.randint(0,4)
    if flip_num==1:
        noise_patch = np.flip(noise_patch, 0).copy()
    if flip_num==2:
        noise_patch = np.flip(noise_patch, 1).copy()
    if flip_num==3:
        noise_patch = np.flip(noise_patch, 2).copy()

    rotate_num = random.randint(0,4)
    if rotate_num==1:
        noise_patch = np.rot90(noise_patch, 1, axes=(1, 2)).copy()
    if rotate_num==2:
        noise_patch = np.rot90(noise_patch, 2, axes=(1, 2)).copy()
    if rotate_num==3:
        noise_patch = np.rot90(noise_patch, 3, axes=(1, 2)).copy()
    '''
    rand_bg = np.random.randint(0, np.max(noise_patch))
    rand_gama_num = random.randint(0,1)
    if rand_gama_num==0:
        rand_gama = np.random.randint(1000, 2000)/1000
    if rand_gama_num==1:
        rand_gama = np.random.randint(500, 1000)/1000
    # print('rand_gama_num shape -----> ',rand_gama_num)
    noise_patch = (noise_patch+rand_bg)/rand_gama
    '''
    return noise_patch


#########################################################################
#########################################################################
class testset_deepsemi(Dataset):
    def __init__(self, img, sub_patch_name_list, single_im_coordinate_list, para_dict):
        self.para_dict = para_dict
        self.sub_patch_name_list = sub_patch_name_list
        self.single_im_coordinate_list = single_im_coordinate_list
        self.img = img

    def __getitem__(self, index):
        single_coordinate = self.single_im_coordinate_list[self.sub_patch_name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        patch_name = self.sub_patch_name_list[index]
        noise_patch = self.img[init_s:end_s,init_h:end_h,init_w:end_w]
        ######################################################
        cut_off = int((self.para_dict['in_c']-1)/2)
        input_patch = realign_patch(noise_patch, self.para_dict['in_c_s'])
        real_A = torch.from_numpy(input_patch).float()
        real_A = real_A.cuda()

        input_patch_t1 = realign_patch_t1(noise_patch, self.para_dict['in_c_t'])
        real_B_t1 = torch.from_numpy(input_patch_t1).float()
        real_B_t1 = real_B_t1.cuda()

        input_patch_t2 = realign_patch_t2(noise_patch, self.para_dict['in_c_t'])
        real_B_t2 = torch.from_numpy(input_patch_t2).float()
        real_B_t2 = real_B_t2.cuda()

        return real_A, real_B_t1, real_B_t2, single_coordinate

    def __len__(self):
        return len(self.sub_patch_name_list)
    

#########################################################################
#########################################################################
def train_preprocess_lessMemory_deepsemi(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s = args.img_s+args.in_c-1
    print('img_s',img_s)

    im_folder = args.datasets_path+'//'+args.datasets_folder
    print('im_folder ',im_folder)
    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = {}

    # print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    img_num_per_stack = math.ceil(args.train_datasets_size/stack_num)
    # print('stack_num -----> ',stack_num)
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print('Preprocess -----> ',im_folder+'//'+im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[0:args.select_img_num,:,:]

        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im).astype(np.float32)/args.norm_factor
        img_list[im_name] = noise_im

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))

        # for ii in range(0, img_num_per_stack):
        while len(name_list)<args.train_datasets_size:
            single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0, 'end_s':0}
            init_h = np.random.randint(0,whole_h-img_h)
            end_h = init_h+img_h
            init_w = np.random.randint(0,whole_w-img_w)
            end_w = init_w+img_w
            init_s = np.random.randint(0,whole_s-img_s)
            end_s = init_s+img_s

            single_coordinate['init_h'] = init_h
            single_coordinate['end_h'] = end_h
            single_coordinate['init_w'] = init_w
            single_coordinate['end_w'] = end_w
            single_coordinate['init_s'] = init_s
            single_coordinate['end_s'] = end_s
            single_coordinate['name'] = im_name

            noise_patch = noise_im[init_s:end_s, init_h:end_h, init_w:end_w]
            noise_patch_mean = np.mean(noise_patch)

            if noise_patch_mean>0:
                patch_name = im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate
    return  name_list, img_list, coordinate_list


#########################################################################
#########################################################################
def test_preprocess_lessMemory_deepsemi(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s+args.in_c-1
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (args.img_s - gap_s2)/2

    print('img_h ',img_h,' img_w ',img_w,' img_s2 ',img_s2,
        ' gap_h ',gap_h,' gap_w ',gap_w,' gap_s2 ',gap_s2,
        ' cut_w ',cut_w,' cut_h ',cut_h,' cut_s ',cut_s,)

    im_folder = args.datasets_path+'//'+args.datasets_folder
    print('im_folder ----> ',im_folder)
    patch_name_list = {}
    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = {}

    print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        name_list.append(im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[-args.select_img_num:,:,:]

        noise_im = (noise_im).astype(np.float32)/args.norm_factor
        img_list[im_name] = noise_im

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]

        num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
        num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
        num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)

        print('num_s ---> ',num_s,'whole_s ---> ',whole_s,'img_s2 ---> ',img_s2,'gap_s2 ---> ',gap_s2)
        single_im_coordinate_list = {}
        sub_patch_name_list = []
        for x in range(0,num_h):
            for y in range(0,num_w):
                for z in range(0,num_s):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    if x != (num_h-1):
                        init_h = gap_h*x
                        end_h = gap_h*x + img_h
                    elif x == (num_h-1):
                        init_h = whole_h - img_h
                        end_h = whole_h

                    if y != (num_w-1):
                        init_w = gap_w*y
                        end_w = gap_w*y + img_w
                    elif y == (num_w-1):
                        init_w = whole_w - img_w
                        end_w = whole_w

                    if z != (num_s-1):
                        init_s = gap_s2*z
                        end_s = gap_s2*z + img_s2 
                    elif z == (num_s-1):
                        init_s = whole_s - img_s2
                        end_s = whole_s 
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s

                    if y == 0:
                        single_coordinate['stack_start_w'] = 0
                        single_coordinate['stack_end_w'] = img_w-cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w-cut_w
                    elif y == num_w-1:
                        single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                        single_coordinate['stack_end_w'] = whole_w
                        single_coordinate['patch_start_w'] = cut_w
                        single_coordinate['patch_end_w'] = img_w
                    else:
                        single_coordinate['stack_start_w'] = y*gap_w+cut_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = cut_w
                        single_coordinate['patch_end_w'] = img_w-cut_w

                    if x == 0:
                        single_coordinate['stack_start_h'] = 0
                        single_coordinate['stack_end_h'] = img_h-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h-cut_h
                    elif x == num_h-1:
                        single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                        single_coordinate['stack_end_h'] = whole_h
                        single_coordinate['patch_start_h'] = cut_h
                        single_coordinate['patch_end_h'] = img_h
                    else:
                        single_coordinate['stack_start_h'] = x*gap_h+cut_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = cut_h
                        single_coordinate['patch_end_h'] = img_h-cut_h

                    if z == 0:
                        single_coordinate['stack_start_s'] = 0
                        single_coordinate['stack_end_s'] = args.img_s-cut_s 
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = args.img_s-cut_s
                        # if single_coordinate['stack_start_h'] == 0 and single_coordinate['stack_start_w'] == 0:
                        #     print('single_coordinate -----> ',single_coordinate)
                    elif z == num_s-1:
                        single_coordinate['stack_start_s'] = whole_s-args.img_s+cut_s -args.in_c+1
                        single_coordinate['stack_end_s'] = whole_s-args.in_c+1
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = args.img_s
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s2+cut_s
                        single_coordinate['stack_end_s'] = z*gap_s2+args.img_s-cut_s
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = args.img_s-cut_s

                    patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
                    sub_patch_name_list.append(patch_name)
                    single_im_coordinate_list[patch_name] = single_coordinate
        coordinate_list[im_name] = single_im_coordinate_list
        patch_name_list[im_name] = sub_patch_name_list
    return  name_list, patch_name_list, img_list, coordinate_list