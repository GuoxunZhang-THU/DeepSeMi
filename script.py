import os
import time

'''
for ii in range(44,45,5):
    os.system('python test_3d_fold2.py --blindspot true \
    --datasets_path ..//datasets --output_dir R_MINST_MOVE_p_dicty \
    --datasets_folder 20221112_T4_tif_c3_MC_event1 \
    --img_h 128 --img_w 128 --img_s 32 --gap_h 96 --gap_w 96 --gap_s 16 \
    --denoise_model SDNN3D3_20221112_T4_tif_c3_MC_event1_lr0.0001_f32_20221125-2118\
    --pth_path pth_bg1_drr10_s3 \
    --input_nc 31 --f_num 32 --normalize_factor 1 --GPU 3 --select_img_num 10000 --pth_index '+str(ii))
'''
train_name = ''
if train_name == '20220408':
    os.system('python train.py --datasets_path datasets \
    --blindspot true --train_datasets_size 2000 --img_h 128 --img_w 128 --img_s 32 --f_num 16 \
    --normalize_factor 1 --lr 0.0001 --n_epochs 100 --GPU 0 --select_img_num 3000 --pth_path pth \
    --input_nc 31 --datasets_folder 20220408')

if train_name == '20220609':
    os.system('python train.py --datasets_path datasets \
    --blindspot true --train_datasets_size 2000 --img_h 128 --img_w 128 --img_s 32 --f_num 16 \
    --normalize_factor 1 --lr 0.0001 --n_epochs 100 --GPU 0 --select_img_num 3000 --pth_path pth \
    --input_nc 31 --datasets_folder 20220609')

if train_name == '20221119':
    os.system('python train.py --datasets_path datasets \
    --blindspot true --train_datasets_size 2000 --img_h 128 --img_w 128 --img_s 32 --f_num 16 \
    --normalize_factor 1 --lr 0.0001 --n_epochs 100 --GPU 0 --select_img_num 3000 --pth_path pth \
    --input_nc 31 --datasets_folder 20221119')



test_name = '20221119'
if test_name == '20220408':
    os.system('python test.py --blindspot true \
    --datasets_path datasets --output_dir results --datasets_folder 20220408 \
    --img_h 128 --img_w 128 --img_s 32 --gap_h 96 --gap_w 96 --gap_s 16 \
    --denoise_model DeepSeMi_20220408 --pth_path pth \
    --input_nc 31 --f_num 16 --normalize_factor 1 --GPU 0 --select_img_num 10000 --pth_index 99')

if test_name == '20220609':
    os.system('python test.py --blindspot true \
    --datasets_path datasets --output_dir results --datasets_folder 20220609 \
    --img_h 128 --img_w 128 --img_s 32 --gap_h 96 --gap_w 96 --gap_s 16 \
    --denoise_model DeepSeMi_20220609 --pth_path pth \
    --input_nc 31 --f_num 16 --normalize_factor 1 --GPU 0 --select_img_num 10000 --pth_index 99')

if test_name == '20221119':
    os.system('python test.py --blindspot true \
    --datasets_path datasets --output_dir results --datasets_folder 20221119 \
    --img_h 128 --img_w 128 --img_s 32 --gap_h 96 --gap_w 96 --gap_s 16 \
    --denoise_model DeepSeMi_20221119 --pth_path pth \
    --input_nc 31 --f_num 16 --normalize_factor 1 --GPU 0 --select_img_num 10000 --pth_index 99')