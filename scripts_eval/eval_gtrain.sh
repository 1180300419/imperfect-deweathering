#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: Liu Xiaohui
 # @Date: 2022-09-16 12:39:08
 # @LastEditors: Liu Xiaohui
 # @LastEditTime: 2022-10-13 23:20:15
### 
echo "Start to test the model...."

name="gt-rain"
device="1"

python test.py \
    --dataset_name GTRAIN\
    --model gtrain\
    --name $name\
    --load_iter 20\
    --calc_metrics True\
    --save_imgs True\
    --gpu_ids $device

