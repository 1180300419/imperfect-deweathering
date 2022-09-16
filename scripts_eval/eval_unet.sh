#!/bin/bash
echo "Start to test the model...."

name="resunet"
dataroot='/home/user/files/data_set/GT-Rain'
device="0"

python test.py \
    --dataset_name GTRAIN\
    --model unet\
    --name $name\
    --load_iter 20
    --dataroot $dataroot\  
    --split 'test'\  
    --save_imgs True\   
    --calc_metrics True\  
    --gpu_id $device\   

