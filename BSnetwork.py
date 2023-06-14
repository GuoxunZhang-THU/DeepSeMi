""" PyTorch implementation of U-Net model for N2N and SSDN.
"""
import torch
import torch.nn as nn
from torch import Tensor
from utility import Shift3d
from utility import rotate3d, rotate3dt
import random
import math

##########################################################################################################################################
##########################################################################################################################################
class BlindspotNetwork_sub(nn.Module):
    def __init__(
        self,
        in_c: int = 3,
        out_c: int = 3,
        blindspot: bool = False,
        f_num: int =32,
        zero_output_weights: bool = False,
        shift_dic = {'in_1_1':0,    'in_1_2':0,
                    'en_1_1':0,    'en_1_2':0,
                    'en_2_1':0,    'en_2_2':0,
                    'en_3_1':0,    'en_3_2':0,
                    'en_6_1':0,    'en_6_2':0,
                    'de_3_1':0,    'de_3_2':0,
                    'de_2_1':0,    'de_2_2':0,
                    'de_1_1':0,    'de_1_2':0,}
    ):
        super(BlindspotNetwork_sub, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights

        if self.blindspot:
            self.ShiftConv3d = ShiftConv3d
            # print('ShiftConv3d')
        else:
            self.Conv3d = nn.Conv3d

        ####################################
        # Encode Blocks
        ####################################
        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            return max_pool

        # Layers: enc_conv0, enc_conv1, pool1
        self.input_block_1 = nn.Sequential(
            self.ShiftConv3d(shift_dic['in_1_1'], in_c, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.ShiftConv3d(shift_dic['in_1_2'], f_num, f_num, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.encode_block_1 = nn.Sequential(
            _max_pool_block(nn.MaxPool3d(2)),
            self.ShiftConv3d(shift_dic['en_1_1'], f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.ShiftConv3d(shift_dic['en_1_2'], f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encode_block_2 = nn.Sequential(
                _max_pool_block(nn.MaxPool3d(2)),
                self.ShiftConv3d(shift_dic['en_2_1'], f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                self.ShiftConv3d(shift_dic['en_2_2'], f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        
        self.encode_block_3 = nn.Sequential(
                _max_pool_block(nn.MaxPool3d(2)),
                self.ShiftConv3d(shift_dic['en_3_1'], f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                self.ShiftConv3d(shift_dic['en_3_2'], f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )

        self.encode_block_6 = nn.Sequential(
            self.ShiftConv3d(shift_dic['en_6_1'], f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.ShiftConv3d(shift_dic['en_6_2'], f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        self.decode_block_3 = nn.Sequential(
                self.ShiftConv3d(shift_dic['de_3_1'], f_num*2, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                self.ShiftConv3d(shift_dic['de_3_2'], f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        self.decode_block_2 = nn.Sequential(
                self.ShiftConv3d(shift_dic['de_2_1'], f_num*2, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                self.ShiftConv3d(shift_dic['de_2_2'], f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        self.decode_block_1 = nn.Sequential(
            self.ShiftConv3d(shift_dic['de_1_1'], f_num*2 , f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.ShiftConv3d(shift_dic['in_1_2'], f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0)
    @property
    def blindspot(self) -> bool:
        return self._blindspot

    def init_weights(self):
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        # x = self.dropout(x)
        depth=3
        if depth==3:
            x = self.input_block_1(x)

            pool1 = self.encode_block_1(x)
            pool2 = self.encode_block_2(pool1)
            pool3 = self.encode_block_3(pool2)
            encoded = self.encode_block_6(pool3)

            # Decoder
            upsample5 = self.decode_block_6(encoded)
            concat3 = torch.cat((upsample5, pool2), dim=1)
            upsample2 = self.decode_block_3(concat3)
            concat2 = torch.cat((upsample2, pool1), dim=1)
            upsample1 = self.decode_block_2(concat2)
            concat1 = torch.cat((upsample1, x), dim=1)
            x = self.decode_block_1(concat1)

        if depth==0:
            x = self.input_block_1(x)
            concat1 = torch.cat((x, x), dim=1)
            x = self.decode_block_1(concat1)
        return x


##########################################################################################################################################
##########################################################################################################################################
class BlindspotNetwork(nn.Module):
    def __init__(
        self,
        in_c_s: int = 3,
        in_c_t: int = 3,
        out_c: int = 3,
        blindspot: bool = True,
        f_num: int =32,
        zero_output_weights: bool = True,
        net_type = 'XY1T1',
        r_num_xy = 4,
        r_num_t = 2,

    ):
        super(BlindspotNetwork, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        self.nin_a_io = 0
        self.net_type = net_type
        #################################
        self.r_num_xy = r_num_xy
        if self.r_num_xy == 4:
            self.r_xy_list     = [0, 90,  180,  270]
            self.inv_r_xy_list = [0, 270, 180,  90]
        if self.r_num_xy == 3:
            self.r_xy_list     = [0, 90,  180]
            self.inv_r_xy_list = [0, 270, 180]
        if self.r_num_xy == 2:
            self.r_xy_list     = [0, 90]
            self.inv_r_xy_list = [0, 270]
        if self.r_num_xy == 1:
            self.r_xy_list     = [0]
            self.inv_r_xy_list = [0]

        self.r_num_t = r_num_t
        if self.r_num_t == 2:
            self.r_t_list     = [90, 270]
            self.inv_r_t_list = [270, 90]
        if self.r_num_t == 1:
            self.r_t_list     = [90]
            self.inv_r_t_list = [270]
        ########################################
        shift_simu_dic = {'in_1_1':0,    'in_1_2':1,
                    'en_1_1':0,    'en_1_2':1,
                    'en_2_1':0,    'en_2_2':1,
                    'en_3_1':0,    'en_3_2':1,
                    'en_6_1':0,    'en_6_2':0,
                    'de_3_1':0,    'de_3_2':0,
                    'de_2_1':0,    'de_2_2':0,
                    'de_1_1':0,    'de_1_2':1,}
        
        shift_1_dic = {'in_1_1':1,    'in_1_2':0,
                    'en_1_1':1,    'en_1_2':0,
                    'en_2_1':1,    'en_2_2':0,
                    'en_3_1':1,    'en_3_2':0,
                    'en_6_1':1,    'en_6_2':0,
                    'de_3_1':0,    'de_3_2':0,
                    'de_2_1':0,    'de_2_2':0,
                    'de_1_1':0,    'de_1_2':0,}

        shift_3_dic = {'in_1_1':1,    'in_1_2':0,
                    'en_1_1':1,    'en_1_2':0,
                    'en_2_1':0,    'en_2_2':0,
                    'en_3_1':0,    'en_3_2':0,
                    'en_6_1':0,    'en_6_2':0,
                    'de_3_1':0,    'de_3_2':0,
                    'de_2_1':0,    'de_2_2':0,
                    'de_1_1':1,    'de_1_2':0,}
 
        shift_5_dic =  {'in_1_1':1,    'in_1_2':1,
                    'en_1_1':1,    'en_1_2':0,
                    'en_2_1':0,    'en_2_2':0,
                    'en_3_1':0,    'en_3_2':0,
                    'en_6_1':0,    'en_6_2':0,
                    'de_3_1':0,    'de_3_2':0,
                    'de_2_1':0,    'de_2_2':0,
                    'de_1_1':0,    'de_1_2':0,} 

        shift_test2_dic =  {'in_1_1':1,    'in_1_2':0,
                            'en_1_1':1,    'en_1_2':0,
                            'en_2_1':1,    'en_2_2':0,
                            'en_3_1':1,    'en_3_2':0,
                            'en_6_1':-1,    'en_6_2':-1,
                            'de_3_1':-1,    'de_3_2':-1,
                            'de_2_1':-1,    'de_2_2':-1,
                            'de_1_1':-1,    'de_1_2':-1,}  

        shift_test1_dic =  {'in_1_1':1,    'in_1_2':1,
                            'en_1_1':1,    'en_1_2':1,
                            'en_2_1':1,    'en_2_2':1,
                            'en_3_1':1,    'en_3_2':1,
                            'en_6_1':-1,    'en_6_2':-1,
                            'de_3_1':-1,    'de_3_2':-1,
                            'de_2_1':-1,    'de_2_2':-1,
                            'de_1_1':-1,    'de_1_2':-1,} 

        shift_N_dic =  {'in_1_1':-1,    'in_1_2':-1,
                        'en_1_1':-1,    'en_1_2':-1,
                        'en_2_1':-1,    'en_2_2':-1,
                        'en_3_1':-1,    'en_3_2':-1,
                        'en_6_1':-1,    'en_6_2':-1,
                        'de_3_1':-1,    'de_3_2':-1,
                        'de_2_1':-1,    'de_2_2':-1,
                        'de_1_1':-1,    'de_1_2':-1,} 

        shift_S_dic =  {'in_1_1':0,    'in_1_2':0,
                        'en_1_1':0,    'en_1_2':0,
                        'en_2_1':0,    'en_2_2':0,
                        'en_3_1':0,    'en_3_2':0,
                        'en_6_1':0,    'en_6_2':0,
                        'de_3_1':0,    'de_3_2':0,
                        'de_2_1':0,    'de_2_2':0,
                        'de_1_1':0,    'de_1_2':0,}   
        ######################################################
        if 'T' in self.net_type:
            if 'Tsim' in self.net_type:
                T_shift_dic = shift_simu_dic
            if 'Ttest2' in self.net_type:
                T_shift_dic = shift_test2_dic
            if 'Ttest1' in self.net_type:
                T_shift_dic = shift_test1_dic
            if 'TN' in self.net_type:
                T_shift_dic = shift_N_dic
            if 'T1' in self.net_type:
                T_shift_dic = shift_1_dic
            if 'T3' in self.net_type:
                T_shift_dic = shift_3_dic 
            if 'T5' in self.net_type:
                T_shift_dic = shift_5_dic
            if 'TS' in self.net_type:
                T_shift_dic = shift_S_dic

            self.Network_T = BlindspotNetwork_sub(in_c = in_c_t,
                                                out_c = out_c,
                                                blindspot = blindspot,
                                                f_num = f_num,
                                                zero_output_weights = zero_output_weights,
                                                shift_dic = T_shift_dic)
            self.nin_a_io = self.nin_a_io+f_num*self.r_num_t

        if 'XY' in self.net_type:
            if 'XYsim' in self.net_type:
                XY_shift_dic = shift_simu_dic
            if 'XYtest2' in self.net_type:
                XY_shift_dic = shift_test2_dic
            if 'XYtest1' in self.net_type:
                XY_shift_dic = shift_test1_dic
            if 'XYN' in self.net_type:
                XY_shift_dic = shift_N_dic
            if 'XY1' in self.net_type:
                XY_shift_dic = shift_1_dic
            if 'XY3' in self.net_type:
                XY_shift_dic = shift_3_dic 
            if 'XY5' in self.net_type:
                XY_shift_dic = shift_5_dic
            if 'XYS' in self.net_type:
                XY_shift_dic = shift_S_dic

            self.Network_XY = BlindspotNetwork_sub(in_c = in_c_s,
                                                out_c = out_c,
                                                blindspot = blindspot,
                                                f_num = f_num,
                                                zero_output_weights = zero_output_weights,
                                                shift_dic = XY_shift_dic)
            self.nin_a_io = self.nin_a_io+f_num*self.r_num_xy
        ####################################
        # Output Block
        ####################################
        '''
        if self.blindspot:
            # Shift 1 pixel down
            self.shift = Shift3d((1, 0, 0))
            # 4 x Channels due to batch rotations
            nin_a_io = f_num*6
        else:
            nin_a_io = f_num*6

        # nin_a,b,c, linear_act
        # self.output_conv = self.Conv3d(f_num, out_channels, 1)
        print('nin_a_io -----> ',nin_a_io)
        self.output_conv = nn.Conv3d(f_num, out_channels, 1)
        '''
        self.output_block = nn.Sequential(
            nn.Conv3d(self.nin_a_io, f_num, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f_num, f_num, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f_num, out_c, 1),
        )

        if 'T' in self.net_type:
            self.output_block_T = nn.Sequential(
                nn.Conv3d(f_num, f_num, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_num, f_num, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_num, out_c, 1),
            )
        if 'XY' in self.net_type:
            self.output_block_XY = nn.Sequential(
                nn.Conv3d(f_num, f_num, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_num, f_num, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(f_num, out_c, 1),
            )
        self.init_weights()

    @property
    def blindspot(self) -> bool:
        return self._blindspot

    def init_weights(self):
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        # Initialise last output layer
        '''
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")
        '''

    def forward(self, x, x_t_in1, x_t_in2: Tensor) -> Tensor:
        sub_x_t_list = []
        sub_x_s_list = []
        shuffle_f = True
        if self.blindspot:
            ##############################################################
            ##############################################################
            if 'T' in self.net_type:
                if self.r_num_t==2:
                    x_t_in1r = rotate3dt(x_t_in1, 90)
                    x_t_in2r = rotate3dt(x_t_in2, 270)
                    x_t = torch.cat((x_t_in1r, x_t_in2r), dim=0)

                    x_t = self.Network_T(x_t)

                    shifted_t = x_t
                    rotated_batch_t = torch.chunk(shifted_t, 2, dim=0)
                    aligned_t = [
                        rotate3dt(rotated_t, rot) for rotated_t, rot in zip(rotated_batch_t, (270, 90))
                    ]
                    if shuffle_f:
                        aligned_t1 = []
                        index_list = list(range(0, len(aligned_t)))
                        random.shuffle(index_list)
                        random_index_list = index_list
                        for aligned_t_i in range(0,len(aligned_t)):
                            aligned_t1.append(aligned_t[random_index_list[aligned_t_i]])
                    if not shuffle_f:
                        aligned_t1 = aligned_t
                    x_t = torch.cat(aligned_t1, dim=1)

                    for i in range(0,len(aligned_t)):
                        aligned_t_s = aligned_t[i]
                        sub_x_t = self.output_block_T(aligned_t_s)
                        sub_x_t_list.append(sub_x_t)

                if self.r_num_t==1:
                    x_t_in1r = rotate3dt(x_t_in1, 90)
                    x_t = self.Network_T(x_t_in1r)
                    x_t = rotate3dt(x_t, 270)

                    aligned_t = x_t
                    sub_x_t = self.output_block_T(aligned_t)
                    sub_x_t_list.append(sub_x_t)


            ##############################################################
            # 0, 90, 180, 270
            # 0, 270, 180, 90
            ##############################################################
            if 'XY' in self.net_type:
                rotated_s = [rotate3d(x, rot) for rot in self.r_xy_list]
                x_s = torch.cat((rotated_s), dim=0)

                x_s = self.Network_XY(x_s)
                shifted_s = x_s
                # Unstack, rotate and combine
                rotated_batch_s = torch.chunk(shifted_s, 4, dim=0)
                aligned_s = [
                    rotate3d(rotated_s, rot) for rotated_s, rot in zip(rotated_batch_s, self.inv_r_xy_list)
                ]
                if shuffle_f:
                    aligned_s1 = []
                    index_list = list(range(0, len(aligned_s)))
                    random.shuffle(index_list)
                    random_index_list = index_list
                    for aligned_s_i in range(0,len(aligned_s)):
                        aligned_s1.append(aligned_s[random_index_list[aligned_s_i]])
                if not shuffle_f:
                    aligned_s1 = aligned_s
                x_s = torch.cat(aligned_s1, dim=1)

        if 'T' in self.net_type and 'XY' in self.net_type:
            # print('x_t, x_s -----> ',x_t.shape, x_s.shape)
            x = torch.cat((x_t, x_s), dim=1)
        if 'T' in self.net_type and 'XY' not in self.net_type:
            x = x_t
        if 'T' not in self.net_type and 'XY' in self.net_type:
            x = x_s
        x = self.output_block(x)

        return x, sub_x_s_list, sub_x_t_list


##############################################################################
##############################################################################
# input B C T Y X
class ShiftConv3d(nn.Conv3d):
    def __init__(self, shift, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # shift = 1
        self.shift_size = (math.ceil(self.kernel_size[1] // 2)+shift, 0, 0)
        # print('self.kernel_size ---> ',self.kernel_size)
        # print('self.shift_size ---> ',self.shift_size)
        # print('self.kernel_size[0] ---> ',self.kernel_size[0])
        shift = Shift3d(self.shift_size)
        self.pad3d = shift.pad3d
        self.crop3d = shift.crop3d

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad3d(x)
        x = super().forward(x)
        x = self.crop3d(x)
        return x