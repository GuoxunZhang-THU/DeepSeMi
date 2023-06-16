import os
import time

train_name = ''
if train_name == '20220408':
    os.system('python train_deepsemi.py \
    --n_epochs 100 --GPU 0 --f_num 8 --batch_size 1 \
    --img_w 144 --img_h 144 --img_s 32 --in_c 15 \
    --datasets_path datasets \
    --datasets_folder 20221119 \
    --pth_path pth \
    --net_type 20220408')

if train_name == '20220609':
    os.system('python train_deepsemi.py \
    --n_epochs 100 --GPU 0 --f_num 8 --batch_size 1 \
    --img_w 144 --img_h 144 --img_s 32 --in_c 15 \
    --datasets_path datasets \
    --datasets_folder 20221119 \
    --pth_path pth \
    --net_type 20220609')

if train_name == '20221119':
    os.system('python train_deepsemi.py \
    --n_epochs 100 --GPU 0 --f_num 8 --batch_size 1 \
    --img_w 144 --img_h 144 --img_s 32 --in_c 15 \
    --datasets_path datasets \
    --datasets_folder 20221119 \
    --pth_path pth \
    --net_type T3XY3')



test_name = '20221119'
if test_name == '20220408':
    os.system('python test_deepsemi.py \
    --GPU 0 --batch_size 2 \
    --img_w 192 --img_h 192 --img_s 32 --in_c 31 \
    --gap_w 160 --gap_h 160 --gap_s 16 \
    --datasets_path datasets \
    --datasets_folder 20220408 \
    --pth_path pth \
    --denoise_model DeepSeMi_20220408 \
    --pth_index deepsemi_99.pth')

if test_name == '20220609':
    os.system('python test_deepsemi.py \
    --GPU 0 --batch_size 2 \
    --img_w 192 --img_h 192 --img_s 32 --in_c 31 \
    --gap_w 160 --gap_h 160 --gap_s 16 \
    --datasets_path datasets \
    --datasets_folder 20220609 \
    --pth_path pth \
    --denoise_model DeepSeMi_20220609 \
    --pth_index deepsemi_40.pth')

if test_name == '20221119':
    os.system('python test_deepsemi.py \
    --GPU 0 --batch_size 2 \
    --img_w 192 --img_h 192 --img_s 32 --in_c 31 \
    --gap_w 160 --gap_h 160 --gap_s 16 \
    --datasets_path datasets \
    --datasets_folder 20221119 \
    --pth_path pth \
    --denoise_model DeepSeMi_20221119 \
    --pth_index deepsemi_99.pth')