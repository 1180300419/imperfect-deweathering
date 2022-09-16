#!/bin/bash

echo "Start to train the model...."

name="fusion_96_0.1mixup_24_middle_blk"
dataroot='/hdd1/MIPI2022/Fusion/dataset/'

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
    --dataset_name FUSION  --model  naf2    --name $name    --dataroot $dataroot  --split 'trainall' --noise_level '_42db'\
    --patch_size 256      --niter 1000       --lr_decay_iters 100   --save_imgs True   --lr 3e-4  --augment True\
    --batch_size 8       --print_freq 10   --calc_metrics True --gpu_ids 5 --mask_size 1  -j 4   | tee $LOG  




