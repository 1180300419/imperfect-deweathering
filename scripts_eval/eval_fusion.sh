#!/bin/bash
echo "Start to test the model...."

name="fusion_64_24_middle_blk_3e-5"
# name="rcan_rawl1rgbl10.05swd"
dataroot='/home'
device="7"

python test.py \
    --dataset_name FUSION  --model naf2test   --name $name         --dataroot $dataroot  --split 'realtest'  --noise_level '_0db'\
    --save_imgs True   --calc_metrics False  --gpu_id $device   --mask_size 1  \
    --load_iter 6
# python evaluate_fusion.py  --name $name --dataroot $dataroot --device $device  --cfa 'GBRG'  --noise_level -1

# python metrics.py --name $name --device 0
