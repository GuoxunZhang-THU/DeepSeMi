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


def shuffle_datasets(train_raw, train_GT, name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    train_raw = np.array(train_raw)
    # print('train_raw shape -----> ',train_raw.shape)
    train_GT = np.array(train_GT)
    # print('train_GT shape -----> ',train_GT.shape)
    new_train_raw = train_raw
    new_train_GT = train_GT
    for i in range(0,len(random_index_list)):
        # print('i -----> ',i)
        new_train_raw[i,:,:,:] = train_raw[random_index_list[i],:,:,:]
        new_train_GT[i,:,:,:] = train_GT[random_index_list[i],:,:,:]
        new_name_list[i] = name_list[random_index_list[i]]
    # new_train_raw = np.expand_dims(new_train_raw, 4)
    # new_train_GT = np.expand_dims(new_train_GT, 4)
    return new_train_raw, new_train_GT, new_name_list


def train_preprocess_lessMemoryMulImages(args,subfolder_name):
    img_h = args.img_h
    img_w = args.img_w

    if subfolder_name == 'trainA':
        im_folder = args.datasets_path+'//'+args.datasets_folder+'//'+args.trainA_folder
    if subfolder_name == 'trainB':
        im_folder = args.datasets_path+'//'+args.datasets_folder+'//'+args.trainB_folder
    name_list = []
    coordinate_list={}
    noise_im_list={}
    # print('im_folder -----> ',im_folder)
    # print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    # print('stack_num -----> ',stack_num)
    size_list = np.zeros((2,stack_num))
    pixel_num_list = np.zeros(stack_num)
    datasize_list = np.zeros(stack_num)
    for i in range(0,len(list(os.walk(im_folder, topdown=False))[-1][-1])):
        im_name = list(os.walk(im_folder, topdown=False))[-1][-1][i]
        # print('read 1 im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        # print('noise_im -----> ',noise_im.shape)
        size_list[0,i] = noise_im.shape[-2]
        size_list[1,i] = noise_im.shape[-1]
        pixel_num_list[i] = noise_im.shape[-1]*noise_im.shape[-2]
    # print('size_list -----> ',size_list)
    # print('pixel_num_list -----> ',pixel_num_list)

    for i in range(0,len(list(os.walk(im_folder, topdown=False))[-1][-1])):
        datasize_list[i] = int(math.ceil(args.train_datasets_size*pixel_num_list[i]/np.sum(pixel_num_list)))
    # print('datasize_list -----> ',datasize_list)

    for i in range(0,len(list(os.walk(im_folder, topdown=False))[-1][-1])):
        im_name = list(os.walk(im_folder, topdown=False))[-1][-1][i]
        # print('read 2 im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im1 = tiff.imread(im_dir)
        noise_im1 = (noise_im1-noise_im1.min()).astype(np.float32)/args.normalize_factor
        # print('noise_im -----> ',noise_im.shape)
        if len(noise_im1.shape)==2:
            noise_im = np.zeros((noise_im1.shape[0],noise_im1.shape[1],3))
            noise_im[:,:,1] = noise_im1
            noise_im[:,:,2] = noise_im1
            noise_im[:,:,0] = noise_im1
        if len(noise_im1.shape)==3:
            noise_im = noise_im1
        # print('noise_im -----> ',noise_im.shape)
        whole_w = noise_im.shape[1]
        whole_h = noise_im.shape[0]
        # print('whole_w -----> ',whole_w)
        # print('whole_h -----> ',whole_h)

        noise_im_list[im_name.replace('.tif','')] = noise_im
        # print('noise_im max ',np.max(noise_im),' noise_im min ',np.min(noise_im))
        # print('im_name.replace() -----> ',im_name.replace('.tif',''))
        # print('datasize_list[i] -----> ',int(datasize_list[i]))
        for ii in range(0, int(datasize_list[i])):
            single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0}
            init_h = np.random.randint(0,whole_h-img_h)
            end_h = init_h+img_h
            init_w = np.random.randint(0,whole_w-img_w)
            end_w = init_w+img_w

            single_coordinate['init_h'] = init_h
            single_coordinate['end_h'] = end_h
            single_coordinate['init_w'] = init_w
            single_coordinate['end_w'] = end_w
            patch_name = im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)
            name_list.append(patch_name)
            coordinate_list[patch_name] = single_coordinate
    return  name_list, noise_im_list, coordinate_list


def shuffle_datasets_lessMemory(name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    for i in range(0,len(random_index_list)):
        new_name_list[i] = name_list[random_index_list[i]]
    return new_name_list


def train_preprocess_lessMemory(args):
    name_list = []
    im_list={}
    im_folder = args.datasets_path+'//'+args.datasets_folder+'//'+args.train_folder
    if len(list(os.walk(im_folder, topdown=False))[-1][-1])<args.train_datasets_size:
        args.train_datasets_size = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    for i in range(0, args.train_datasets_size):
        im_name = list(os.walk(im_folder, topdown=False))[-1][-1][i]
        # print('read im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        im = tiff.imread(im_dir)
        # im = im.transpose(2,0,1)
        # print(im.shape)
        if len(im.shape)==2:
            im = np.expand_dims(im, axis=0)
        if im.shape[0]>3:
            im = im.transpose(2,0,1)
        im = im.astype(np.float32)/args.normalize_factor

        im_list[im_name.replace('.tif','')] = im
        name_list.append(im_name.replace('.tif',''))
    return  name_list, im_list



def test_preprocess_lessMemory(args):
    name_list = []
    im_list={}
    im_folder = args.datasets_path+'//'+args.datasets_folder+'//'+args.test_folder
    if len(list(os.walk(im_folder, topdown=False))[-1][-1])<args.test_datasets_size:
        args.test_datasets_size = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    for i in range(0, args.test_datasets_size):
        im_name = list(os.walk(im_folder, topdown=False))[-1][-1][i]
        # print('read im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        im = tiff.imread(im_dir)
        # im = im.transpose(2,0,1)
        # print(im.shape)
        if len(im.shape)==2:
            im = np.expand_dims(im, axis=0)
        if im.shape[0]>3:
            im = im.transpose(2,0,1)
        im = im.astype(np.float32)/args.normalize_factor

        im_list[im_name.replace('.tif','')] = im
        name_list.append(im_name.replace('.tif',''))
    return  name_list, im_list


def train_preprocess_lessMemoryMulStacksRand(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s = args.img_s
    print('img_s',img_s)

    im_folder = args.datasets_path+'//'+args.datasets_folder
    print('im_folder ',im_folder)
    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = {}

    print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    img_num_per_stack = math.ceil(args.train_datasets_size/stack_num)
    print('stack_num -----> ',stack_num)
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[0:args.select_img_num,:,:]

        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im).astype(np.float32)/args.normalize_factor
        img_list[im_name] = noise_im

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for ii in range(0, img_num_per_stack):
            single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0, 'end_s':0}
            init_h = np.random.randint(0,whole_h-img_h)
            end_h = init_h+img_h
            init_w = np.random.randint(0,whole_w-img_w)
            end_w = init_w+img_w
            init_s = np.random.randint(10,whole_s-img_s-10)
            end_s = init_s+img_s

            single_coordinate['init_h'] = init_h
            single_coordinate['end_h'] = end_h
            single_coordinate['init_w'] = init_w
            single_coordinate['end_w'] = end_w
            single_coordinate['init_s'] = init_s
            single_coordinate['end_s'] = end_s
            single_coordinate['name'] = im_name

            patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
            # train_raw.append(noise_patch1.transpose(1,2,0))
            name_list.append(patch_name)
            # print(' single_coordinate -----> ',single_coordinate)
            coordinate_list[patch_name] = single_coordinate
    return  name_list, img_list, coordinate_list


def train_preprocess_lessMemoryMulStacksRandFold(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s = args.img_s+args.input_nc-1
    print('img_s',img_s)

    im_folder = args.datasets_path+'//'+args.datasets_folder
    print('im_folder ',im_folder)
    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = {}

    print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    img_num_per_stack = math.ceil(args.train_datasets_size/stack_num)
    print('stack_num -----> ',stack_num)
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print('im_name -----> ',im_folder+'//'+im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[0:args.select_img_num,:,:]

        print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im).astype(np.float32)/args.normalize_factor
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
                patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate
    return  name_list, img_list, coordinate_list


def test_preprocess_lessMemory_MC(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s2 - gap_s2)/2
    '''
    print('img_h ',img_h,' img_w ',img_w,' img_s2 ',img_s2,
        ' gap_h ',gap_h,' gap_w ',gap_w,' gap_s2 ',gap_s2,
        ' cut_w ',cut_w,' cut_h ',cut_h,' cut_s ',cut_s,)
    '''
    im_folder = args.datasets_path+'//'+args.datasets_folder
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
            noise_im = noise_im[0:args.select_img_num,:,:]

        noise_im = (noise_im).astype(np.float32)/args.normalize_factor
        img_list[im_name] = noise_im

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]

        num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
        num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
        num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)

        # print('num_s ---> ',num_s,'whole_s ---> ',whole_s,'img_s2 ---> ',img_s2,'gap_s2 ---> ',gap_s2)
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
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
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
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
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
                        single_coordinate['stack_start_s'] = z*gap_s2 
                        single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s 
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s2-cut_s
                    elif z == num_s-1:
                        single_coordinate['stack_start_s'] = whole_s-img_s2+cut_s 
                        single_coordinate['stack_end_s'] = whole_s 
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = img_s2
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s2+cut_s
                        single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = img_s2-cut_s

                    patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
                    sub_patch_name_list.append(patch_name)
                    single_im_coordinate_list[patch_name] = single_coordinate
        coordinate_list[im_name] = single_im_coordinate_list
        patch_name_list[im_name] = sub_patch_name_list
    return  name_list, patch_name_list, img_list, coordinate_list



def test_preprocess_lessMemory_MCfold(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s+args.input_nc-1
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

        noise_im = (noise_im).astype(np.float32)/args.normalize_factor
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
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
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
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
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
                        single_coordinate['stack_start_s'] = z*gap_s2 
                        single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s 
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = args.img_s-cut_s
                    elif z == num_s-1:
                        single_coordinate['stack_start_s'] = whole_s-img_s2+cut_s 
                        single_coordinate['stack_end_s'] = whole_s 
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = args.img_s
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s2+cut_s
                        single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = args.img_s-cut_s

                    patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
                    sub_patch_name_list.append(patch_name)
                    single_im_coordinate_list[patch_name] = single_coordinate
        coordinate_list[im_name] = single_im_coordinate_list
        patch_name_list[im_name] = sub_patch_name_list
    return  name_list, patch_name_list, img_list, coordinate_list