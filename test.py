import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import sys
import math
import itertools
import scipy.io as scio

import numpy as np
from data_process import shuffle_datasets, test_preprocess_lessMemory_MCfold, shuffle_datasets_lessMemory
from utils import save_tiff_image, save_feature_tiff_image
from skimage import io

# from noise_network_3d_T import NoiseNetwork_3D
from noise_network_3d import NoiseNetwork_3D
# from noise_network_3d_mix import NoiseNetwork_3D

import random
#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--GPU', type=int, default=3, help="the index of GPU you will use for computation")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--output_dir', type=str, default='./results', help="the output folder")
parser.add_argument('--batch_size', type=int, default=1, help="size of the batchs")

parser.add_argument('--img_w', type=int, default=512, help="")
parser.add_argument('--img_h', type=int, default=512, help="")
parser.add_argument('--img_s', type=int, default=512, help="")
parser.add_argument('--gap_w', type=int, default=512, help="")
parser.add_argument('--gap_h', type=int, default=512, help="")
parser.add_argument('--gap_s', type=int, default=512, help="")

parser.add_argument('--input_nc', type=int, default=1, help="")
parser.add_argument('--input_nc_s', type=int, default=1, help="")
parser.add_argument('--input_nc_t', type=int, default=1, help="")

parser.add_argument('--output_nc', type=int, default=1, help="")
parser.add_argument('--f_num', type=int, default=1, help="")

parser.add_argument('--normalize_factor', type=int, default=10000, help='actions: train or predict')

parser.add_argument('--datasets_folder', type=str, default='rawdata', help="the name of your project")
parser.add_argument('--datasets_path', type=str, default='datasets', help="the name of your project")
parser.add_argument('--test_folder', type=str, default='train_raw', help="")
parser.add_argument('--pth_path', type=str, default='pth', help="the name of your project")

parser.add_argument('--denoise_model', type=str, default='ModelForPytorch', help='A')
parser.add_argument('--pth_index', type=int, default=16, help='the height of image gap')

parser.add_argument('--train_datasets_size', type=int, default=1000, help='actions: train or predict')

parser.add_argument('--blindspot', type=bool, default=True, help='')
parser.add_argument('--zero_output_weights', type=bool, default=False, help='')
parser.add_argument('--select_img_num', type=int, default=6000, help='select the number of images')

opt = parser.parse_args()
print('the parameter of your training ----->')
print(opt)

opt.input_nc_s = opt.input_nc
opt.input_nc_t = int((opt.input_nc+1)/2)
########################################################################################################################
if not os.path.exists(opt.output_dir): 
    os.mkdir(opt.output_dir)
current_time = 'R_'+opt.datasets_folder+'_'+opt.denoise_model

output_path1 = opt.output_dir + '//' + current_time
output_path = output_path1+'//'+str(opt.pth_index)

if not os.path.exists(output_path1): 
    os.mkdir(output_path1)
if not os.path.exists(output_path): 
    os.mkdir(output_path)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)

im_name_list, patch_name_list, img_list, coordinate_list = \
test_preprocess_lessMemory_MCfold(opt)

'''
net = NoiseNetwork_3D(in_channels = opt.input_nc ,
                    out_channels = opt.output_nc ,
                    blindspot = opt.blindspot,
                    f_num = opt.f_num)
'''

net = NoiseNetwork_3D(in_channels_s = opt.input_nc_s ,
                    in_channels_t = opt.input_nc_t ,
                    out_channels = opt.output_nc ,
                    blindspot = opt.blindspot,
                    f_num = opt.f_num)

net.cuda()
net_pth_name = 'A_'+str(opt.pth_index)+'.pth'
net.load_state_dict(torch.load(opt.pth_path+'//'+opt.denoise_model+'//'+net_pth_name))
########################################################################################################################
def realign_patch(patch, in_nc):
    re_patch = np.zeros((in_nc, patch.shape[0]-in_nc+1, patch.shape[1], patch.shape[2]))
    for i in range(0, patch.shape[0]-in_nc+1):
        re_patch[:,i,:,:] = patch[i:i+in_nc,:,:]
    return re_patch

def realign_patch_t1(patch, in_nc):
    re_patch = np.zeros((in_nc, patch.shape[0]-in_nc*2+2, patch.shape[1], patch.shape[2]))
    # print(patch.shape[0]-in_nc*2+2)
    for i in range(0, patch.shape[0]-in_nc*2+2):
        re_patch[:,i,:,:] = patch[i:i+in_nc,:,:]
    # print('re_patch -----> ',re_patch.shape)
    return re_patch

def realign_patch_t2(patch, in_nc):
    re_patch = np.zeros((in_nc, patch.shape[0]-in_nc*2+2, patch.shape[1], patch.shape[2]))
    # print(in_nc-1, patch.shape[0]-in_nc+1)
    # print('re_patch -----> ',re_patch.shape)
    for i in range(in_nc-1, patch.shape[0]-in_nc+1):
        re_patch[:,i-in_nc+1,:,:] = patch[i:i+in_nc,:,:]
    return re_patch
########################################################################################################################
prev_time = time.time()
time_start=time.time()

cut_off = int((opt.input_nc-1)/2)
for index in range(len(im_name_list)): # 1):# 
    im_name = im_name_list[index]  #'minst_4.tif' # 'dicty CV membrane Z004_2_24_326_c1_s7.tif' #
    single_im_coordinate_list = coordinate_list[im_name]

    noise_img = img_list[im_name]
    
    denoise_img = np.zeros(noise_img.shape)
    input_img = np.zeros(noise_img.shape)
    sub_patch_name_list = patch_name_list[im_name]
    for subindex in range(len(sub_patch_name_list)):
        single_coordinate = single_im_coordinate_list[sub_patch_name_list[subindex]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        patch_name = sub_patch_name_list[subindex]
        noise_patch = noise_img[init_s:end_s,init_h:end_h,init_w:end_w]
        ################################################################################################################
        # noise_patch1 = noise_img[init_s+1:end_s:2, init_h:end_h, init_w:end_w]
        input_patch = realign_patch(noise_patch, opt.input_nc_s)
        real_A = torch.from_numpy(np.expand_dims(input_patch,0)).float()
        real_A = real_A.cuda()

        input_patch_t1 = realign_patch_t1(noise_patch, opt.input_nc_t)
        real_B_t1 = torch.from_numpy(np.expand_dims(input_patch_t1,0)).float()
        real_B_t1 = real_B_t1.cuda()

        input_patch_t2 = realign_patch_t2(noise_patch, opt.input_nc_t)
        real_B_t2 = torch.from_numpy(np.expand_dims(input_patch_t2,0)).float()
        real_B_t2 = real_B_t2.cuda()
        # print(real_A.shape,' --- ',real_B.shape)
        fake_A ,sub_x_s_list,  aaaaa = net(real_A, real_B_t1, real_B_t2)

        output_image = fake_A.cpu().detach().numpy().squeeze().astype(np.float32)*opt.normalize_factor
        ################################################################################################################
        # Determine approximate time left
        batches_done = index
        batches_left = 1 * len(sub_patch_name_list) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        prev_time = time.time()
        ################################################################################################################
        if index%1 == 0:
            time_end=time.time()
            time_cost=datetime.timedelta(seconds= (time_end - time_start))
            sys.stdout.write("\r [Batch %d/%d] [Time Left: %s] [Time Cost: %s]"
            % (subindex,
            len(sub_patch_name_list),
            time_left,
            time_cost,))
        ################################################################################################################
        stack_start_w = int(single_coordinate['stack_start_w'])
        stack_end_w = int(single_coordinate['stack_end_w'])
        patch_start_w = int(single_coordinate['patch_start_w'])
        patch_end_w = int(single_coordinate['patch_end_w'])

        stack_start_h = int(single_coordinate['stack_start_h'])
        stack_end_h = int(single_coordinate['stack_end_h'])
        patch_start_h = int(single_coordinate['patch_start_h'])
        patch_end_h = int(single_coordinate['patch_end_h'])

        stack_start_s = int(single_coordinate['stack_start_s'])+cut_off
        stack_end_s = int(single_coordinate['stack_end_s'])-cut_off
        patch_start_s = int(single_coordinate['patch_start_s'])
        patch_end_s = int(single_coordinate['patch_end_s'])
        # print(stack_start_s,' --- ',stack_end_s,' --- ',cut_off)
        # print(patch_start_s,' --- ',patch_end_s,' --- ',cut_off)
        if len(output_image.shape)==3:
            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
            = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w] 

        if len(output_image.shape)==2:
            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
            = output_image[patch_start_h:patch_end_h, patch_start_w:patch_end_w] 
        '''
        output_name = output_path + '//' +im_name.replace('.tif','')+'_b'+str(opt.b_lstm_pth)+ '_f'\
            +str(opt.f_lstm_pth)+ '_u'+str(opt.u_net_pth)+ '_'+str(index)+ '_'+str(subindex)+ '_output.tif'
        io.imsave(output_name, output_image)
        '''
    # del noise_img np.float32 np.uint16
    denoise_img = denoise_img/np.max(denoise_img)*65535
    denoise_img = denoise_img.squeeze().astype(np.float32)*opt.normalize_factor

    # denoise_img = np.clip(denoise_img, 0, 65535).astype('uint16')
    result_name = output_path + '//' +im_name.replace('.tif','')+'_'+str(opt.pth_index)+ '_output.tif'
    io.imsave(result_name, denoise_img)