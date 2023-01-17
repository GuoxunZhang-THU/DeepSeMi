""" PyTorch implementation of U-Net model for N2N and SSDN.
"""
import torch
import torch.nn as nn
from torch import Tensor
from utility import Shift3d
from utility import rotate3d, rotate3dt
import random
import math

## blind spot 3-3
class NoiseNetwork_sub(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        blindspot: bool = False,
        f_num: int =32,
        zero_output_weights: bool = False,
    ):
        super(NoiseNetwork_sub, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        # self.Conv3d = ShiftConv3d if self.blindspot else nn.Conv3d
        if self.blindspot:
            self.Conv3d1 = ShiftConv3d1
            self.Conv3d0 = ShiftConv3d0
            print('ShiftConv3d')
        else:
            self.Conv3d = nn.Conv3d
        ####################################
        # Encode Blocks
        ####################################

        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            # if blindspot:
            #     return nn.Sequential(Shift3d((1, 0, 0)), max_pool)
            return max_pool

        # Layers: enc_conv0, enc_conv1, pool1
        self.input_block_1 = nn.Sequential(
            self.Conv3d1(in_channels, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv3d0(f_num, f_num, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        
        self.encode_block_1 = nn.Sequential(
            _max_pool_block(nn.MaxPool3d(2)),
            self.Conv3d1(f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv3d0(f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                _max_pool_block(nn.MaxPool3d(2)),
                self.Conv3d1(f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                self.Conv3d0(f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            self.Conv3d1(f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv3d0(f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            self.Conv3d0(f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv3d0(f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                self.Conv3d0(f_num*2, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                self.Conv3d0(f_num, f_num, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,   +f_num
        self.decode_block_1 = nn.Sequential(
            self.Conv3d0(f_num , f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv3d0(f_num, f_num, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
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
        x = self.dropout(x)
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

        x = self.decode_block_1(x)
        return x
##########################################################################################################################################

class NoiseNetwork_3D(nn.Module):
    def __init__(
        self,
        in_channels_s: int = 3,
        in_channels_t: int = 3,
        out_channels: int = 3,
        blindspot: bool = False,
        f_num: int =32,
        zero_output_weights: bool = False,
    ):
        super(NoiseNetwork_3D, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights

        self.Network_T = NoiseNetwork_sub(in_channels = in_channels_t,
                                            out_channels = out_channels,
                                            blindspot = blindspot,
                                            f_num = f_num,
                                            zero_output_weights = zero_output_weights)

        self.Network_S = NoiseNetwork_sub(in_channels = in_channels_s,
                                            out_channels = out_channels,
                                            blindspot = blindspot,
                                            f_num = f_num,
                                            zero_output_weights = zero_output_weights)

        ####################################
        # Output Block
        ####################################

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
        self.output_block = nn.Sequential(
            nn.Conv3d(nin_a_io, f_num, 1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(f_num, f_num, 1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv,
        )

        self.output_block_T = nn.Sequential(
            nn.Conv3d(f_num, f_num, 1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(f_num, f_num, 1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(f_num, out_channels, 1),
        )

        self.output_block_S = nn.Sequential(
            nn.Conv3d(f_num, f_num, 1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(f_num, f_num, 1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(f_num, out_channels, 1),
        )

        # Initialize weights
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
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x, x_t_in1, x_t_in2: Tensor) -> Tensor:
        if self.blindspot:
            # rotated_t = [rotate3dt(x_t_in, rot) for rot in (90,270)]
            # x_t = torch.cat((rotated_t), dim=0)
            x_t_in1r = rotate3dt(x_t_in1, 90)
            x_t_in2r = rotate3dt(x_t_in2, 270)
            x_t = torch.cat((x_t_in1r, x_t_in2r), dim=0)

            rotated_s = [rotate3d(x, rot) for rot in (0, 90, 180, 270)]
            x_s = torch.cat((rotated_s), dim=0)

        x_t = self.Network_T(x_t)
        x_s = self.Network_S(x_s)

        # Output
        if self.blindspot:
            # Apply shift
            # shifted_t = self.shift(x_t)
            shifted_t = x_t
            # Unstack, rotate and combine
            rotated_batch_t = torch.chunk(shifted_t, 2, dim=0)
            aligned_t = [
                rotate3dt(rotated_t, rot) for rotated_t, rot in zip(rotated_batch_t, (270, 90))
            ]
            # print('aligned_t -----> ',len(aligned_t))
            # print('aligned_t -----> ',aligned_t.shape)
            shuffle_f = True
            if shuffle_f:
                aligned_t1 = []
                index_list = list(range(0, len(aligned_t)))
                # print('index_list -----> ',index_list)
                random.shuffle(index_list)
                random_index_list = index_list
                # print('random_index_list -----> ',random_index_list)
                for aligned_t_i in range(0,len(aligned_t)):
                    # print('random_index_list[aligned_t_i] -----> ',random_index_list[aligned_t_i])
                    aligned_t1.append(aligned_t[random_index_list[aligned_t_i]])
            if not shuffle_f:
                aligned_t1 = aligned_t
            x_t = torch.cat(aligned_t1, dim=1)


            # shifted_s = self.shift(x_s)
            shifted_s = x_s
            # Unstack, rotate and combine
            rotated_batch_s = torch.chunk(shifted_s, 4, dim=0)
            aligned_s = [
                rotate3d(rotated_s, rot) for rotated_s, rot in zip(rotated_batch_s, (0, 270, 180, 90))
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

            # print('x_t -----> ',x_t.shape)
            # print('x_s -----> ',x_s.shape)
            x = torch.cat((x_t, x_s), dim=1)

        x = self.output_block(x)

        sub_x_s_list = []
        for i in range(0,len(aligned_s)):
            aligned_s_s = aligned_s[i]
            sub_x_s = self.output_block_S(aligned_s_s)
            sub_x_s_list.append(sub_x_s)

        sub_x_t_list = []
        for i in range(0,len(aligned_t)):
            aligned_t_s = aligned_t[i]
            sub_x_t = self.output_block_T(aligned_t_s)
            sub_x_t_list.append(sub_x_t)

        return x, sub_x_s_list, sub_x_t_list



# input B C T Y X
class ShiftConv3d0(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super().__init__(*args, **kwargs)
        compen = 0
        self.shift_size = (math.ceil(self.kernel_size[1] // 2)+compen, 0, 0)
        print('self.kernel_size ---> ',self.kernel_size)
        print('self.shift_size ---> ',self.shift_size)
        # print('self.kernel_size[0] ---> ',self.kernel_size[0])
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift3d(self.shift_size)
        self.pad3d = shift.pad3d
        self.crop3d = shift.crop3d

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad3d(x)
        x = super().forward(x)
        x = self.crop3d(x)
        return x



# input B C T Y X
class ShiftConv3d1(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super().__init__(*args, **kwargs)
        compen = 1
        self.shift_size = (math.ceil(self.kernel_size[1] // 2)+compen, 0, 0)
        print('self.kernel_size ---> ',self.kernel_size)
        print('self.shift_size ---> ',self.shift_size)
        # print('self.kernel_size[0] ---> ',self.kernel_size[0])
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift3d(self.shift_size)
        self.pad3d = shift.pad3d
        self.crop3d = shift.crop3d

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad3d(x)
        x = super().forward(x)
        x = self.crop3d(x)
        return x