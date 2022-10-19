#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: Liu Xiaohui
 # @Date: 2022-09-16 12:39:08
 # @LastEditors: Liu Xiaohui
 # @LastEditTime: 2022-10-19 10:20:30
### 
echo "Start to test the model...."

name="align"
device="0"

python test.py \
    --dataset_name GTRAINVAL\
    --model alignednet\
    --name $name\
    --load_iter 20\
    --calc_metrics True\
    --save_imgs True\
    --gpu_ids $device

