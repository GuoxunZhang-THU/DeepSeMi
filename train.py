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
from data_process import shuffle_datasets, train_preprocess_lessMemoryMulStacksRandFold, shuffle_datasets_lessMemory
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
parser.add_argument('--input_nc', type=int, default=1, help="")
parser.add_argument('--input_nc_s', type=int, default=1, help="")
parser.add_argument('--input_nc_t', type=int, default=1, help="")
parser.add_argument('--output_nc', type=int, default=1, help="")
parser.add_argument('--f_num', type=int, default=1, help="")

parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--normalize_factor', type=int, default=10000, help='actions: train or predict')

parser.add_argument('--datasets_folder', type=str, default='rawdata', help="the name of your project")
parser.add_argument('--datasets_path', type=str, default='datasets', help="the name of your project")
parser.add_argument('--train_folder', type=str, default='train_raw', help="")
parser.add_argument('--pth_path', type=str, default='pth', help="the name of your project")
parser.add_argument('--train_datasets_size', type=int, default=1000, help='actions: train or predict')

parser.add_argument('--blindspot', type=bool, default=True, help='')
parser.add_argument('--zero_output_weights', type=bool, default=False, help='')
parser.add_argument('--select_img_num', type=int, default=6000, help='select the number of images')

parser.add_argument('--noise_rate', type=float, default=0.2, help='add noise rate')
parser.add_argument('--mask_sample', type=str, default='grid', help="the name of your project")
parser.add_argument('--mask_mode', type=str, default='zero', help="the name of your project")
opt = parser.parse_args()
print('the parameter of your training ----->')
print(opt)
opt.input_nc_s = opt.input_nc
opt.input_nc_t = int((opt.input_nc+1)/2)
########################################################################################################################
if not os.path.exists(opt.output_dir): 
    os.mkdir(opt.output_dir)
current_time = 'DeepSeMi_'+opt.datasets_folder+'_lr'+str(opt.lr)+'_f'+str(opt.f_num)+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")

output_path = opt.output_dir + '//' + current_time
pth_folder = opt.pth_path+'//'+ current_time

if not os.path.exists(output_path): 
    os.mkdir(output_path)
if not os.path.exists(opt.pth_path): 
    os.mkdir(opt.pth_path)
if not os.path.exists(pth_folder): 
    os.mkdir(pth_folder)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
batch_size = opt.batch_size
lr = opt.lr

name_list, img_list, coordinate_list = train_preprocess_lessMemoryMulStacksRandFold(opt)

net = NoiseNetwork_3D(in_channels_s = opt.input_nc_s ,
                    in_channels_t = opt.input_nc_t ,
                    out_channels = opt.output_nc ,
                    blindspot = opt.blindspot,
                    f_num = opt.f_num)

L1loss_function = torch.nn.L1Loss()
L2loss_function = torch.nn.MSELoss()
if torch.cuda.is_available():
    print('Using GPU.')
    net.cuda()
    L1loss_function.cuda()
    L2loss_function.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, 0.999))
########################################################################################################################
time_start=time.time()
prev_time = time.time()

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
#######################################################################################################################
iteration_num = 0
cut_off = int((opt.input_nc-1)/2)
for epoch in range(0, opt.n_epochs):
    name_list = shuffle_datasets_lessMemory(name_list)
    # print('name list -----> ',name_list)   
    for index in range(len(name_list)):
        patch_name = name_list[index]
        single_coordinate = coordinate_list[patch_name]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        img_name = single_coordinate['name']
        noise_img = img_list[img_name]
        noise_patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]

        # noise_patch1 = noise_img[init_s+1:end_s:2, init_h:end_h, init_w:end_w]
        input_patch = realign_patch(noise_patch, opt.input_nc_s)
        real_A = torch.from_numpy(np.expand_dims(input_patch,0)).float()
        real_A = real_A.cuda()

        GT_patch = noise_patch[cut_off:-cut_off,:,:]
        real_B = torch.from_numpy(np.expand_dims(np.expand_dims(GT_patch, 0),0)).float()
        real_B = real_B.cuda()

        input_patch_t1 = realign_patch_t1(noise_patch, opt.input_nc_t)
        real_B_t1 = torch.from_numpy(np.expand_dims(input_patch_t1,0)).float()
        real_B_t1 = real_B_t1.cuda()

        input_patch_t2 = realign_patch_t2(noise_patch, opt.input_nc_t)
        real_B_t2 = torch.from_numpy(np.expand_dims(input_patch_t2,0)).float()
        real_B_t2 = real_B_t2.cuda()

        fake_A ,sub_x_s_list, sub_x_t_list= net(real_A, real_B_t1, real_B_t2)
        ###############################################################################################################
        optimizer.zero_grad()
        # L1Loss_A2B = L1loss_function(train_imB, pred_imA)
        L1Loss_B2A = L1loss_function(real_B, fake_A)
        L2Loss_B2A = L2loss_function(real_B, fake_A)
        loss = L2Loss_B2A+L1Loss_B2A

        for sub_x_s_i in range(len(sub_x_s_list)):
            sub_x_s = sub_x_s_list[sub_x_s_i]
            loss = loss+L1loss_function(real_B, sub_x_s)+L2loss_function(real_B, sub_x_s)

        for sub_x_t_i in range(len(sub_x_t_list)):
            sub_x_t = sub_x_t_list[sub_x_t_i]
            loss = loss+L1loss_function(real_B, sub_x_t)+L2loss_function(real_B, sub_x_t)

        loss.backward()
        optimizer.step()

        iteration_num = iteration_num +1
        ################################################################################################################
        batches_done = epoch * len(name_list) + index
        batches_left = opt.n_epochs * len(name_list) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        ################################################################################################################ 8_HCC20
        if (index%100 == 0):
            time_end=time.time()
            # print('time cost',time_end-time_start,'s')
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d]  ETA: %s"
            % (epoch, opt.n_epochs, index, len(name_list), time_left ))

            # print('\n')
            # print('    Loss ',loss.cpu().detach().numpy())
            # print('iteration_num ',iteration_num)

        # if (index%50 == 0): # or ((epoch+1)%1 == 0):
        if (iteration_num+1)%50 == 0:
            # print('save image')
            image_name = patch_name
            '''
            real_B_t1_path = output_path + '//real_B_t1'
            if not os.path.exists(real_B_t1_path): 
                os.mkdir(real_B_t1_path)
            real_B_t1 = real_B_t1.cpu().detach().numpy()
            real_B_t1 = real_B_t1.squeeze().astype(np.float32)*opt.normalize_factor
            real_B_t1_name = real_B_t1_path + '//' + str(epoch) + '_' + str(index) + '_' + image_name+'_real_B_t1.tif'
            io.imsave(real_B_t1_name, real_B_t1)

            real_B_t2_path = output_path + '//real_B_t2'
            if not os.path.exists(real_B_t2_path): 
                os.mkdir(real_B_t2_path)
            real_B_t2 = real_B_t2.cpu().detach().numpy()
            real_B_t2 = real_B_t2.squeeze().astype(np.float32)*opt.normalize_factor
            real_B_t2_name = real_B_t2_path + '//' + str(epoch) + '_' + str(index) + '_' + image_name+'_real_B_t2.tif'
            io.imsave(real_B_t2_name, real_B_t2)
            
            real_A_path = output_path + '//real_A'
            if not os.path.exists(real_A_path): 
                os.mkdir(real_A_path)
            real_A = real_A.cpu().detach().numpy()
            real_A = real_A.squeeze().astype(np.float32)*opt.normalize_factor
            real_A_name = real_A_path + '//' + str(epoch) + '_' + str(index) + '_' + image_name+'_real_A.tif'
            io.imsave(real_A_name, real_A)
            '''
            fake_A_path = output_path + '//fake_A'
            if not os.path.exists(fake_A_path): 
                os.mkdir(fake_A_path)
            fake_A = fake_A.cpu().detach().numpy()
            fake_A = fake_A.squeeze().astype(np.float32)*opt.normalize_factor
            fake_A_name = fake_A_path + '//' + str(epoch) + '_' + str(index) + '_' + image_name+'_fake_A.tif'
            io.imsave(fake_A_name, fake_A)


    if (epoch+1)%1 == 0:
        torch.save(net.state_dict(), pth_folder + '//A_' + str(epoch) + '.pth')





