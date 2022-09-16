#!/bin/bash

echo "Start to train the model...."

name="unet"
dataroot='/home/user/files/data_set/GT-Rain'

build_dir="../checkpoints/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=$build_dir/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
        --name $name\
        --dataroot $dataroot\
        --dataset_name GTRAIN\
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
        --gpu_ids 0,1\
        -j 4  | tee $LOG  




