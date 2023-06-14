import torch.nn as nn
from torch import Tensor
from typing import Tuple
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from data_format import DataFormat
from data_format import DataDim
from data_format import DATA_FORMAT_DIM_INDEX


class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.

    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.

    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)


def rotate(x: torch.Tensor, angle: int, data_format: str = DataFormat.BCHW) -> torch.Tensor:
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    # dims = DATA_FORMAT_DIM_INDEX[data_format]
    # h_dim = dims[DataDim.HEIGHT]
    # w_dim = dims[DataDim.WIDTH]
    h_dim = 2
    w_dim = 3

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")


class DataFormat:
    BHWC = "BHWC"
    BWHC = "BWHC"
    BCHW = "BCHW"
    BCWH = "BCWH"
    HWC = "HWC"
    WHC = "WHC"
    CHW = "CHW"
    CWH = "CWH"


################################################################################################################################
class Crop3d(nn.Module):
    def __init__(self, crop: Tuple[int, int, int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 6

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom, front, back) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        t0, t1 = front, x.shape[-3] - back
        # print('t0, t1, y0, y1, x0, x1 ---> ', t0, t1, y0, y1, x0, x1)
        return x[:, :, t0:t1, y0:y1, x0:x1]


class Shift3d(nn.Module):
    # X Y T
    def __init__(self, shift: Tuple[int, int, int]):
        super().__init__()
        self.shift = shift
        # print('self.shift ---> ',self.shift)
        horz, vert, time = self.shift
        # d5, d4, d3 = self.shift
        t_a, t_b = abs(time), 0
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        '''
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        '''
        # Order : Left, Right, Top Bottom
        # print('t_a, t_b, x_a, x_b, y_a, y_b ---> ',t_a, t_b, x_a, x_b, y_a, y_b)
        self.pad3d = nn.ConstantPad3d((x_a, x_b, y_a, y_b, t_a, t_b), 0)
        self.crop3d = Crop3d((x_b, x_a, y_b, y_a, t_b, t_a))
        self.shift_block = nn.Sequential(self.pad3d, self.crop3d)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)


'''
input:,,T,Y,X
input = torch.randn(1, 1, 3, 3, 3)
print('input ---> ',input.shape)
shift_size = (1, 0, 0)
shift = Shift3d(shift_size)

pad = shift.pad3d
crop = shift.crop3d

print('input ---> \n',input)
input_pad = pad(input)
print('input_pad ---> ',input_pad.shape)
print('input_pad ---> \n',input_pad)
crop_pad = crop(input_pad)
print('crop_pad ---> ',crop_pad.shape)
print('crop_pad ---> \n',crop_pad)
'''
'''
m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
output = m(input)
print('output ---> ',output.shape)
'''

def rotate3d(x: torch.Tensor, angle: int) -> torch.Tensor:
    # dims = DATA_FORMAT_DIM_INDEX[data_format]
    # h_dim = dims[DataDim.HEIGHT]
    # w_dim = dims[DataDim.WIDTH]
    h_dim = 3
    w_dim = 4

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")


def rotate3dt(x: torch.Tensor, angle: int) -> torch.Tensor:
    # dims = DATA_FORMAT_DIM_INDEX[data_format]
    # h_dim = dims[DataDim.HEIGHT]
    # w_dim = dims[DataDim.WIDTH]
    t_dim = 2
    h_dim = 3
    w_dim = 4

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(t_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(t_dim)
    elif angle == 270:
        return x.flip(t_dim).transpose(t_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")

class DataFormat3D:
    BCTHW = "BCTHW"


'''
input = torch.randn(1, 1, 2, 3, 4)
input90 = rotate3d(input, 90)
print('input90 ---> ',input90.shape)

input180 = rotate3d(input, 180)
print('input180 ---> ',input180.shape)

input270 = rotate3d(input, 270)
print('input270 ---> ',input270.shape)
'''