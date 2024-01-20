#!/bin/bash

###
 # @Descripttion: 
 # @version: 
 # @Author: Liu Xiaohui
 # @Date: 2022-09-16 10:40:30
 # @LastEditors: Liu Xiaohui
 # @LastEditTime: 2023-06-13 14:06:40
### 

echo "Start to train the model..."

name="loss_0_1_ablation"

build_dir="../checkpoints/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=$build_dir/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
        --train_dataset_size 'all'\
        --val_dataset_size '30'\
        --test_dataset_size 'all'\
        --input_frames 5\
        --dataset_name MULGTWEA\
        --init_type 'xavier'\
        --name $name\
        --dataroot ''\
        --split 'train'\
        --batch_size 8\
        --patch_size 256\
        --model multiencgtrainselfsu\
        --niter 20\
        --lr_policy 'warmup'\
        --lr 2e-4\
        --min_lr 1e-6\
        --warmup_niter 4\
        --save_imgs True\
        --print_freq 100\
        --calc_metrics True\
        --gpu_ids 5\
        -j 4  | tee $LOG  



