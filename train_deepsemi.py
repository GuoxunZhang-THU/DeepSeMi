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
from data_process import train_preprocess_lessMemory_deepsemi, trainset_deepsemi
from utils import save_tiff_image, save_feature_tiff_image
from utils import save_img_train, save_para_dict, UseStyle, get_netpara
from skimage import io

from BSnetwork import BlindspotNetwork

import random
import warnings
warnings.filterwarnings("ignore")
import yaml
import sys
import scipy.io as scio
#############################################################################################################################################
#############################################################################################################################################
class train_deepsemi():
    def __init__(self, train_para):
        self.n_epochs = 100
        self.GPU = '0'
        self.output_dir = './results'
        self.batch_size = 2

        self.img_w = 128
        self.img_h = 128
        self.img_s = 128

        self.in_c = 17
        self.in_c_s = 128
        self.in_c_t = 128
        self.out_c = 1
        self.f_num = 8

        self.lr = 0.0001
        self.b1 = 0.5
        self.b2 = 0.999
        self.norm_factor = 1

        self.use_pretrain = 0
        self.pretrain_path = ''
        self.pretrain_model = 'datasets'
        self.pretrain_index = ''
        self.blindspot = True

        self.datasets_folder = ''
        self.datasets_path = 'datasets'
        self.pth_path = 'pth'
        self.train_datasets_size = 1000
        self.select_img_num = 5000
        self.net_type = 'T3XY3'

        self.reset_para(train_para)
        self.make_folder()
        self.in_c_s = self.in_c
        self.in_c_t = int((self.in_c+1)/2)


    #########################################################################
    #########################################################################
    def make_folder(self): 
        current_time = 'DeepSeMi_'+self.net_type+'_'+self.datasets_folder[0:10]+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.output_path = self.output_dir + '/' + current_time
        if not os.path.exists(self.output_dir): 
            os.mkdir(self.output_dir)
        if not os.path.exists(self.output_path): 
            os.mkdir(self.output_path)

        self.pth_save_path = self.pth_path+'//'+ current_time
        if not os.path.exists(self.pth_path): 
            os.mkdir(self.pth_path)
        if not os.path.exists(self.pth_save_path): 
            os.mkdir(self.pth_save_path)


    #########################################################################
    #########################################################################
    def reset_para(self, para):
        for key, value in para.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.use_pretrain:
            pretrain_yaml = self.pretrain_path+'//'+self.pretrain_model+'//'+'DeepSeMi_para.yaml'
            print('pretrain_yaml -----> ', pretrain_yaml, os.path.exists(pretrain_yaml))
            if os.path.exists(pretrain_yaml):
                # with open(pretrain_yaml, 'r') as f:
                f = open(pretrain_yaml)
                pretrain_para = yaml.load(f.read(), Loader = yaml.FullLoader)
                setattr(self, 'f_num', pretrain_para['f_num'])
                setattr(self, 'in_c', pretrain_para['in_c'])
                setattr(self, 'out_c', pretrain_para['out_c'])
                setattr(self, 'norm_factor', pretrain_para['norm_factor'])
                # setattr(self, 'net_type', pretrain_para['net_type'])
                print('##### Pretrain net type:'+pretrain_para['net_type'],
                      ' ##### Now net type:'+self.net_type)
        print(UseStyle('Training parameters ----->', mode = 'bold', fore  = 'red'))
        print(self.__dict__)


    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        del yaml_dict['DeepSeMi_net'] 
        del yaml_dict['optimizer'] 
        yaml_name = 'DeepSeMi_para.yaml'
        save_para_path = self.output_path+ '//'+yaml_name
        save_para_dict(save_para_path, yaml_dict)
        save_para_path = self.pth_save_path+ '//'+yaml_name
        save_para_dict(save_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'DeepSeMi_para_'+self.datasets_folder+str(get_netpara(self.DeepSeMi_net))+'.txt'
        save_para_path = self.output_path+ '//'+txt_name
        save_para_dict(save_para_path, txt_dict)
        save_para_path = self.pth_save_path+ '//'+txt_name
        save_para_dict(save_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list #str(opt.GPU)
        self.DeepSeMi_net = BlindspotNetwork(in_c_s = self.in_c_s ,
                                            in_c_t = self.in_c_t ,
                                            out_c = self.out_c ,
                                            blindspot = self.blindspot,
                                            f_num = self.f_num,
                                            net_type = self.net_type)
        self.DeepSeMi_net = torch.nn.DataParallel(self.DeepSeMi_net) 
        self.DeepSeMi_net.cuda()

        if self.use_pretrain:
            net_pth_name = self.pretrain_index
            net_model_path = self.pretrain_path+'//'+self.pretrain_model+'//'+net_pth_name
            self.DeepSeMi_net.load_state_dict(torch.load(net_model_path))

        self.optimizer = torch.optim.Adam(self.DeepSeMi_net.parameters(),
                                        lr=self.lr, betas=(self.b1, self.b2))
        print('Parameters of DeepSeMi_net -----> ' , get_netpara(self.DeepSeMi_net) )
        self.save_para()


    #########################################################################
    #########################################################################
    def generate_patch(self):
        self.name_list, self.img_list, self.coordinate_list = train_preprocess_lessMemory_deepsemi(self)


    #########################################################################
    #########################################################################
    def train(self):
        loss_m = []

        Tensor = torch.cuda.FloatTensor
        per_epoch_len = len(self.name_list)
        self.L1loss_function = torch.nn.L1Loss()
        self.L2loss_function = torch.nn.MSELoss()

        self.L1loss_function.cuda()
        self.L2loss_function.cuda()

        prev_time = time.time()
        ########################################################################################################################
        torch.multiprocessing.set_start_method('spawn')
        ########################################################################################################################
        time_start=time.time()
        for epoch in range(0, self.n_epochs):
            train_data = trainset_deepsemi(self.name_list, self.img_list, self.coordinate_list, self.__dict__)
            trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
            for index, (real_A, real_B, real_B_t1, real_B_t2, patch_name) in enumerate(trainloader):
                # print(' real_A ---> ',real_A.shape,' real_B ---> ',real_B.shape,' real_B_t1 ---> ',real_B_t1.shape,' real_B_t2 ---> ',real_B_t2.shape)
                fake_A ,sub_x_s_list, sub_x_t_list= self.DeepSeMi_net(real_A, real_B_t1, real_B_t2)

                self.optimizer.zero_grad()
                # L1Loss_A2B = L1loss_function(train_imB, pred_imA)
                L1Loss_B2A = self.L1loss_function(real_B, fake_A)
                L2Loss_B2A = self.L2loss_function(real_B, fake_A)
                loss = L2Loss_B2A+L1Loss_B2A

                real_B_save = real_B[ :, :, int(real_B.shape[2]*0.25):int(real_B.shape[2]*0.75), int(real_B.shape[3]*0.25):int(real_B.shape[3]*0.75), int(real_B.shape[4]*0.25):int(real_B.shape[4]*0.75), ]
                fake_A_save = fake_A[ :, :, int(fake_A.shape[2]*0.25):int(fake_A.shape[2]*0.75), int(fake_A.shape[3]*0.25):int(fake_A.shape[3]*0.75), int(fake_A.shape[4]*0.25):int(fake_A.shape[4]*0.75), ]
                loss_save = self.L1loss_function(real_B_save, fake_A_save) + self.L2loss_function(real_B_save, fake_A_save)
                loss_m.append(loss_save.cpu().detach().numpy())

                for sub_x_s_i in range(len(sub_x_s_list)):
                    sub_x_s = sub_x_s_list[sub_x_s_i]
                    loss = loss + self.L1loss_function(real_B, sub_x_s) + self.L2loss_function(real_B, sub_x_s)

                for sub_x_t_i in range(len(sub_x_t_list)):
                    sub_x_t = sub_x_t_list[sub_x_t_i]
                    loss = loss + self.L1loss_function(real_B, sub_x_t) + self.L2loss_function(real_B, sub_x_t)

                loss.backward()
                self.optimizer.step()
                ################################################################################################################
                batches_done = epoch * per_epoch_len + index + 1
                batches_left = self.n_epochs * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left/batches_done * (time.time() - prev_time)))
                # prev_time = time.time()
                ################################################################################################################
                if index%(100//self.batch_size) == 0:
                    time_end=time.time()
                    print_head = 'DeepSeMi_TRAIN'
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '    [Epoch %d/%d]   [Batch %d/%d]   [Total_loss: %f]   [Time_Left: %s]'% (
                        epoch,
                        self.n_epochs,
                        index,
                        per_epoch_len,
                        loss, 
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                ################################################################################################################
                if (index+1)%(500//self.batch_size) == 0:
                    norm_factor = self.norm_factor
                    image_name = patch_name

                    save_img_train(real_B, self.output_path, epoch, index, image_name, norm_factor, 'real_B')
                    save_img_train(fake_A, self.output_path, epoch, index, image_name, norm_factor, 'fake_A')

            torch.save(self.DeepSeMi_net.state_dict(), self.pth_save_path +'//'+'deepsemi_' + str(epoch) + '.pth')

            loss_save_path = self.pth_save_path +'//'+'loss_'+str(per_epoch_len)+ '.mat'
            data_loss = {}
            data_loss['loss'] = loss_m
            scio.savemat(loss_save_path, {'loss':data_loss['loss']})


    #########################################################################
    #########################################################################
    def run(self):
        self.initialize_model()
        self.generate_patch()
        self.train()


if __name__ == '__main__':
    #########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=str, default='1', help=" ")
    parser.add_argument("--n_epochs", type=int, default=100, help=" ")
    parser.add_argument("--batch_size", type=int, default=1, help=" ")
    parser.add_argument("--output_dir", type=str, default='./results', help=" ")

    parser.add_argument("--img_w", type=int, default=128, help=" ")
    parser.add_argument("--img_h", type=int, default=128, help=" ")
    parser.add_argument("--img_s", type=int, default=32, help=" ")

    parser.add_argument("--in_c", type=int, default=17, help=" ")
    parser.add_argument("--in_c_s", type=int, default=128, help=" ")
    parser.add_argument("--in_c_t", type=int, default=32, help=" ")
    parser.add_argument("--out_c", type=int, default=1, help=" ")
    parser.add_argument("--f_num", type=int, default=8, help=" ")

    parser.add_argument('--lr', type=float, default=0.0001, help=" ")
    parser.add_argument('--b1', type=float, default=0.5, help=" ")
    parser.add_argument('--b2', type=float, default=0.999, help=" ")
    parser.add_argument("--norm_factor", type=int, default=1, help=" ")

    parser.add_argument("--use_pretrain", type=int, default=0, help=" ")
    parser.add_argument("--pretrain_path", type=str, default='', help=" ")
    parser.add_argument("--pretrain_model", type=str, default='datasets', help=" ")
    parser.add_argument("--pretrain_index", type=str, default='', help=" ")
    parser.add_argument('--blindspot', type=bool, default=True, help='')

    parser.add_argument("--datasets_folder", type=str, default='20220408', help=" ")
    parser.add_argument("--datasets_path", type=str, default='..//datasets', help=" ")
    parser.add_argument("--pth_path", type=str, default='pth', help=" ")

    parser.add_argument("--train_datasets_size", type=int, default=1000, help=" ")
    parser.add_argument("--select_img_num", type=int, default=5000, help=" ")
    parser.add_argument("--net_type", type=str, default='T1XY1', help=" ")
    # net_type: T1XY1 or T3XY3 or T5XY5

    opt = parser.parse_args()
    print('the parameter of your training ----->')
    print(opt)
    #########################################################################
    train_parameters={'GPU':'1',
                    'n_epochs':100,
                    'batch_size':1,
                    'output_dir':'./results',
                    
                    'img_w':128,
                    'img_h':128,
                    'img_s':32,
                    
                    'in_c':17,
                    'in_c_s':128,
                    'in_c_t':32,
                    'out_c':1,
                    'f_num':8,
                    
                    'lr':0.0001,
                    'b1':0.5,
                    'b2':0.999,
                    'norm_factor':1,
                    
                    'use_pretrain':0,
                    'pretrain_path':'',
                    'pretrain_model':'datasets',
                    'pretrain_index':'',
                    'blindspot':True,

                    'datasets_folder':'20220408',
                    'datasets_path':'..//datasets',
                    'pth_path':'pth',
                    
                    'train_datasets_size':1000,
                    'select_img_num':5000,
                    'net_type':'XYS',}
    #########################################################################
    train_parameters['GPU'] = opt.GPU
    train_parameters['n_epochs'] = opt.n_epochs
    train_parameters['batch_size'] = opt.batch_size
    train_parameters['output_dir'] = opt.output_dir

    train_parameters['img_w'] = opt.img_w
    train_parameters['img_h'] = opt.img_h
    train_parameters['img_s'] = opt.img_s

    train_parameters['in_c'] = opt.in_c
    train_parameters['in_c_s'] = opt.in_c_s
    train_parameters['in_c_t'] = opt.in_c_t
    train_parameters['out_c'] = opt.out_c
    train_parameters['f_num'] = opt.f_num  

    train_parameters['in_c_s'] = opt.in_c_s
    train_parameters['in_c_t'] = opt.in_c_t
    train_parameters['out_c'] = opt.out_c
    train_parameters['f_num'] = opt.f_num  

    train_parameters['lr'] = opt.lr
    train_parameters['b1'] = opt.b1
    train_parameters['b2'] = opt.b2
    train_parameters['norm_factor'] = opt.norm_factor    

    train_parameters['use_pretrain'] = opt.use_pretrain
    train_parameters['pretrain_path'] = opt.pretrain_path
    train_parameters['pretrain_model'] = opt.pretrain_model
    train_parameters['pretrain_index'] = opt.pretrain_index
    train_parameters['blindspot'] = opt.blindspot  

    train_parameters['datasets_folder'] = opt.datasets_folder
    train_parameters['datasets_path'] = opt.datasets_path
    train_parameters['pth_path'] = opt.pth_path            

    train_parameters['train_datasets_size'] = opt.train_datasets_size
    train_parameters['select_img_num'] = opt.select_img_num
    train_parameters['net_type'] = opt.net_type     
    #########################################################################
    deepsemi_model = train_deepsemi(train_parameters)
    deepsemi_model.run()






