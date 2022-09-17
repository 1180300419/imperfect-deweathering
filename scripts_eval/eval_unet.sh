#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: Liu Xiaohui
 # @Date: 2022-09-16 12:39:08
 # @LastEditors: Liu Xiaohui
 # @LastEditTime: 2022-09-17 13:13:01
### 
echo "Start to test the model...."

name="unet"
device="0"

python test.py \
    --dataset_name GTRAIN\
    --model unet\
    --name $name\
    --load_iter 19\
    --calc_metrics True\
    --save_imgs True\
    --gpu_ids $device

