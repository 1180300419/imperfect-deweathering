#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: Liu Xiaohui
 # @Date: 2022-09-16 12:39:08
 # @LastEditors: Liu Xiaohui
 # @LastEditTime: 2022-10-18 20:25:19
### 
# echo "Start to test the model...."

name="unet-4resblocks-2e-4_rainaug"
device="1"

python test_internet_img.py \
    --dataset_name TEST\
    --model unet\
    --name $name\
    --load_iter 20\
    --calc_metrics True\
    --save_imgs True\
    --gpu_ids $device\
    --test_internet True

