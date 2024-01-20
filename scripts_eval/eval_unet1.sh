echo "Start to test the model...."

name="try_l1_01swd-varient_gtrain_snow"
device="4"
load_iter=18
build_dir="../checkpoints/"$name"/test_epoch_"$load_iter

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=$build_dir/`date +%Y-%m-%d-%H-%M-%S`.txt

python test.py \
    --test_dataset_size 'all'\
    --input_frames 1\
    --dataset_name MULGTWEA\
    --model multiencgtrainselfsu\
    --load_iter $load_iter\
    --name $name\
    --calc_metrics True\
    --save_imgs True\
    --gpu_ids $device\
    -j 4 | tee $LOG


