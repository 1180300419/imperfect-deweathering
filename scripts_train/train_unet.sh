#!/bin/bash

###
 # @Descripttion: 
 # @version: 
 # @Author: Liu Xiaohui
 # @Date: 2022-09-16 10:40:30
 # @LastEditors: Liu Xiaohui
 # @LastEditTime: 2022-09-16 13:33:36
### 

echo "Start to train the model..."

name="unet-laplacian"

build_dir="../checkpoints/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=$build_dir/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
        --dataset_name GTRAIN\
        --name $name\
        --dataroot ''\
        --split 'train'\
        --batch_size 8\
        --patch_size 256\
        --model  unet\
        --niter 20\
        --lr_policy 'warmup'\
        --lr 2e-4\
        --min_lr 1e-6\
        --warmup_niter 4\
        --save_imgs True\
        --print_freq 100\
        --calc_metrics True\
        --gpu_ids 0\
        -j 4  | tee $LOG  




