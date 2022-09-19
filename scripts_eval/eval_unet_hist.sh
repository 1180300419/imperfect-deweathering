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

name="unet-hist"
device="1"

python test.py \
    --dataset_name GTRAIN\
    --model unet\
    --name $name\
    --load_iter 39\
    --calc_metrics True\
    --save_imgs True\
    --gpu_ids $device

