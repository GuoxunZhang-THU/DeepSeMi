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
from data_process import test_preprocess_lessMemory_deepsemi, testset_deepsemi
from utils import save_tiff_image, save_feature_tiff_image
from skimage import io

from BSnetwork import BlindspotNetwork

import random
import warnings
warnings.filterwarnings("ignore")
import yaml
from utils import save_img_train, save_para_dict, UseStyle, save_img, get_netpara
#############################################################################################################################################

class test_deepsemi():
    def __init__(self, test_para):
        self.GPU = '0'
        self.output_dir = './results'
        self.batch_size = 2

        self.img_w = 128
        self.img_h = 128
        self.img_s = 128
        self.gap_w = 128
        self.gap_h = 128
        self.gap_s = 128

        self.in_c = 17
        self.in_c_s = 128
        self.in_c_t = 128
        self.out_c = 1
        self.f_num = 8

        self.norm_factor = 1
        self.datasets_folder = ''
        self.datasets_path = 'datasets'
        self.net_type = 'T3XY3'

        self.select_img_num = 5000
        self.pth_path = 'pth'
        self.denoise_model = 'pth'
        self.pth_index = 'pth'

        self.reset_para(test_para)
        self.make_folder()
        self.in_c_s = self.in_c
        self.in_c_t = int((self.in_c+1)/2)
        # gap_rate = 0.4
        # self.gap_w = int(self.img_w*gap_rate)
        # self.gap_h = int(self.img_h*gap_rate)
        # self.gap_s = int(self.img_s*gap_rate)


    #########################################################################
    #########################################################################
    def make_folder(self):
        current_time = self.denoise_model+'_'+self.datasets_folder
        self.output_path = self.output_dir + '//' + current_time.replace('//','_')
        if not os.path.exists(self.output_dir): 
            os.mkdir(self.output_dir)
        if not os.path.exists(self.output_path): 
            os.mkdir(self.output_path)


    #########################################################################
    #########################################################################
    def reset_para(self, para):
        for key, value in para.items():
            if hasattr(self, key):
                setattr(self, key, value)

        yaml_path = self.pth_path+'//'+self.denoise_model+'//'+'DeepSeMi_para.yaml'
        with open(yaml_path, "r") as yaml_file:
            saved_para = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # print('siganl_para -----> ',siganl_para)
        self.norm_factor = saved_para['norm_factor']
        self.f_num = saved_para['f_num']
        self.in_c = saved_para['in_c']
        self.out_c = saved_para['out_c']
        self.net_type = saved_para['net_type']

        print(UseStyle('Training parameters ----->', mode = 'bold', fore  = 'red'))
        print(self.__dict__)


    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        del yaml_dict['DeepSeMi_net'] 
        yaml_name = 'DeepSeMi_para.yaml'
        save_para_path = self.output_path+ '//'+yaml_name
        save_para_dict(save_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'DeepSeMi_para_'+str(get_netpara(self.DeepSeMi_net))+'.txt'
        save_para_path = self.output_path+ '//'+txt_name
        save_para_dict(save_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list #str(opt.GPU)
        self.DeepSeMi_net = BlindspotNetwork(in_c_s = self.in_c_s ,
                                            in_c_t = self.in_c_t ,
                                            out_c = self.out_c ,
                                            blindspot = True,
                                            f_num = self.f_num,
                                            net_type = self.net_type)
        self.DeepSeMi_net = torch.nn.DataParallel(self.DeepSeMi_net) 
        self.DeepSeMi_net.cuda()

        pth_name = self.pth_index
        model_path = self.pth_path+'//'+self.denoise_model+'//'+pth_name
        self.DeepSeMi_net.load_state_dict(torch.load(model_path))
        print('Parameters of DeepSeMi_net -----> ' , get_netpara(self.DeepSeMi_net) )
        self.save_para()


    #########################################################################
    #########################################################################
    def generate_patch(self):
        # self.name_list, self.img_list, self.coordinate_list = train_preprocess_lessMemory_deepsemi(self)
        self.im_name_list, self.patch_name_list, self.img_list, self.coordinate_list = test_preprocess_lessMemory_deepsemi(self)


    #########################################################################
    #########################################################################
    def test(self):
        torch.multiprocessing.set_start_method('spawn')
        prev_time = time.time()
        iteration_num = 0
        cut_off = int((self.in_c-1)/2)
        for index in range(len(self.im_name_list)): 
            im_name = self.im_name_list[index]  
            single_im_coordinate_list = self.coordinate_list[im_name]
            noise_img = self.img_list[im_name]
            sub_patch_name_list = self.patch_name_list[im_name]
            
            denoise_img = np.zeros(noise_img.shape)
            ######################################################################################
            test_deepsemi_data = testset_deepsemi(noise_img, sub_patch_name_list, single_im_coordinate_list, self.__dict__)
            test_deepsemi_dataloader = DataLoader(test_deepsemi_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
            ######################################################################################
            for iteration, (real_A, real_B_t1, real_B_t2, single_coordinate) in enumerate(test_deepsemi_dataloader): 
                fake_A, aaaaaa,  aaaaa = self.DeepSeMi_net(real_A, real_B_t1, real_B_t2)
                ################################################################################################################
                per_epoch_len = len(single_im_coordinate_list)//self.batch_size
                batches_done = index * per_epoch_len + iteration + 1
                batches_left = len(self.im_name_list) * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left/batches_done * (time.time() - prev_time)))
                # prev_time = time.time()
                ################################################################################################################
                if iteration%(1) == 0:
                    time_end=time.time()
                    print_head = 'DeepSeMi_TEST'
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '   [Batch %d/%d]   [Time_Left: %s]'% (
                        batches_done,
                        batches_left+batches_done,
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                deepsemi_out = fake_A.cpu().detach().numpy().squeeze().astype(np.float32)
                # print('deepsemi_out -----> ',deepsemi_out.shape)
                if len(deepsemi_out.shape)==3:
                    deepsemi_out = deepsemi_out[np.newaxis,:,:,:]
                # print('deepsemi_out -----> ',deepsemi_out.shape)
                for deepsemi_i in range(0, deepsemi_out.shape[0]):
                    deepsemi_out_s = np.squeeze(deepsemi_out[deepsemi_i,:,:,:])

                    stack_start_w = int(single_coordinate['stack_start_w'][deepsemi_i])
                    stack_end_w = int(single_coordinate['stack_end_w'][deepsemi_i])
                    patch_start_w = int(single_coordinate['patch_start_w'][deepsemi_i])
                    patch_end_w = int(single_coordinate['patch_end_w'][deepsemi_i])

                    stack_start_h = int(single_coordinate['stack_start_h'][deepsemi_i])
                    stack_end_h = int(single_coordinate['stack_end_h'][deepsemi_i])
                    patch_start_h = int(single_coordinate['patch_start_h'][deepsemi_i])
                    patch_end_h = int(single_coordinate['patch_end_h'][deepsemi_i])

                    stack_start_s = int(single_coordinate['stack_start_s'][deepsemi_i])
                    stack_end_s = int(single_coordinate['stack_end_s'][deepsemi_i])
                    patch_start_s = int(single_coordinate['patch_start_s'][deepsemi_i])
                    patch_end_s = int(single_coordinate['patch_end_s'][deepsemi_i])
                    # print(single_coordinate)
                    # print('-----> ',stack_start_s,stack_end_s,stack_start_h,stack_end_h,stack_start_w,stack_end_w)
                    # print('-----> ',patch_start_s,patch_end_s,patch_start_h,patch_end_h,patch_start_w,patch_end_w)
                    denoise_img[cut_off+stack_start_s:cut_off+stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = deepsemi_out_s[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w] 

            save_img(denoise_img, self.norm_factor, self.output_path, im_name)


    #########################################################################
    #########################################################################
    def run(self):
        self.initialize_model()
        self.generate_patch()
        self.test()


if __name__ == '__main__':
    #########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=str, default='1', help=" ")
    parser.add_argument("--batch_size", type=int, default=1, help=" ")
    parser.add_argument("--output_dir", type=str, default='./results', help=" ")

    parser.add_argument("--img_w", type=int, default=128, help=" ")
    parser.add_argument("--img_h", type=int, default=128, help=" ")
    parser.add_argument("--img_s", type=int, default=32, help=" ")
    parser.add_argument("--gap_w", type=int, default=128, help=" ")
    parser.add_argument("--gap_h", type=int, default=128, help=" ")
    parser.add_argument("--gap_s", type=int, default=32, help=" ")

    parser.add_argument("--in_c", type=int, default=17, help=" ")
    parser.add_argument("--in_c_s", type=int, default=128, help=" ")
    parser.add_argument("--in_c_t", type=int, default=32, help=" ")
    parser.add_argument("--out_c", type=int, default=1, help=" ")
    parser.add_argument("--f_num", type=int, default=8, help=" ")

    parser.add_argument("--norm_factor", type=int, default=1, help=" ")
    parser.add_argument("--net_type", type=str, default='XYS', help=" ")
    parser.add_argument("--datasets_folder", type=str, default='20220408', help=" ")
    parser.add_argument("--datasets_path", type=str, default='..//datasets', help=" ")

    parser.add_argument("--select_img_num", type=int, default=5000, help=" ")
    parser.add_argument("--pth_path", type=str, default='', help=" ")
    parser.add_argument("--denoise_model", type=str, default='datasets', help=" ")
    parser.add_argument("--pth_index", type=str, default='', help=" ")
    opt = parser.parse_args()
    print('the parameter of your training ----->')
    print(opt)
    #########################################################################
    test_parameters = { 'GPU' : '0,1',
                    'output_dir' : './/test_results',
                    'batch_size' : 2,
                    ###########################
                    'img_w' : 128,
                    'img_h' : 128,
                    'img_s' : 128,
                    'gap_w' : 2,
                    'gap_h' : 2,
                    'gap_s' : 2,
                    ###########################
                    'in_c' : 17,
                    'in_c_s' : 128,
                    'in_c_t' : 128,
                    'out_c' : 1,
                    'f_num' : 8,
                    ###########################
                    'norm_factor' : 1,
                    'datasets_folder' : '',
                    'datasets_path' : 'datasets',
                    'net_type' : 'T3XY3',
                    ###########################
                    'select_img_num' : 5000,
                    'pth_path' : 'pth',
                    'denoise_model' : 'pth',
                    'pth_index' : 'pth',
    }
    #########################################################################
    test_parameters['GPU'] = opt.GPU
    test_parameters['output_dir'] = opt.output_dir
    test_parameters['batch_size'] = opt.batch_size

    test_parameters['img_w'] = opt.img_w
    test_parameters['img_h'] = opt.img_h
    test_parameters['img_s'] = opt.img_s
    test_parameters['gap_w'] = opt.gap_w
    test_parameters['gap_h'] = opt.gap_h
    test_parameters['gap_s'] = opt.gap_s

    test_parameters['in_c'] = opt.in_c
    test_parameters['in_c_s'] = opt.in_c_s
    test_parameters['in_c_t'] = opt.in_c_t
    test_parameters['out_c'] = opt.out_c
    test_parameters['f_num'] = opt.f_num

    test_parameters['norm_factor'] = opt.norm_factor
    test_parameters['datasets_folder'] = opt.datasets_folder
    test_parameters['datasets_path'] = opt.datasets_path
    test_parameters['net_type'] = opt.net_type

    test_parameters['select_img_num'] = opt.select_img_num
    test_parameters['pth_path'] = opt.pth_path
    test_parameters['denoise_model'] = opt.denoise_model
    test_parameters['pth_index'] = opt.pth_index
    #########################################################################
    deepsemi_model = test_deepsemi(test_parameters)
    deepsemi_model.run()