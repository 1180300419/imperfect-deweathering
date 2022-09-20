#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: Liu Xiaohui
 # @Date: 2022-09-16 12:39:08
 # @LastEditors: Liu Xiaohui
 # @LastEditTime: 2022-09-17 16:32:17
### 
echo "Start to test the model...."

name="gt-rain-unet-guidedfilter"
device="7"

python test_internet_img.py \
    --dataset_name TEST\
    --model unet\
    --name $name\
    --load_iter 40\
    --calc_metrics False\
    --save_imgs True\
    --gpu_ids $device

