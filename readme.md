# Learning Real-World Image De-Weathering with Imperfect Supervision
[![arXiv](https://img.shields.io/badge/arXiv-2305.13077-b31b1b.svg)](https://arxiv.org/abs/2310.14958)

## News
- **Oct 24, 2023:** The paper is released!

<hr />

> **Abstract:** *Real-world image de-weathering aims at removing various undesirable weather-related artifacts. Owing to the impossibility of capturing image pairs concurrently, existing real-world de-weathering datasets often exhibit inconsistent illumination, position, and textures between the ground-truth images and the input degraded images, resulting in imperfect supervision. Such non-ideal supervision negatively affects the training process of learning-based de-weathering methods. In this work, we attempt to address the problem with a unified solution for various inconsistencies. Specifically, inspired by information bottleneck theory, we first develop a Consistent Label Constructor (CLC) to generate a pseudo-label as consistent as possible with the input degraded image while removing most weather-related degradations. In particular, multiple adjacent frames of the current input are also fed into CLC to enhance the pseudo-label. Then we combine the original imperfect labels and pseudo-labels to jointly supervise the de-weathering model by the proposed Information Allocation Strategy (IAS). During testing, only the de-weathering model is used for inference. Experiments on two real-world de-weathering datasets show that our method helps existing de-weathering models achieve better performance.* 
<hr />

## Setup

### 1. Prepare Datasets
<details>
<summary><b>GT-Rain dataset</b></summary>
Download the dataset from [here](https://drive.google.com/drive/folders/1NSRl954QPcGIgoyJa_VjQwh_gEaHWPb8).
</details>

<details>
<summary><b>WeatherStream dataset</b></summary>
Download the dataset from [here](https://drive.google.com/drive/folders/12Z9rBSTs0PPNHLieyU2vnCTzR6fOFLrT).
</details>

### 2. Download Weights
The pretrained model of "Ours-RainRobust" trained using GT-Rain-Snow and WeatherStream can be downloaded from [url1](https://pan.baidu.com/s/1C2cSg6pfInEQOGMM4ro53w?pwd=rs5g) (password:rs5g) and [url2](https://pan.baidu.com/s/14ROb7g1NbPmaM2bBgmA0Vw?pwd=r3g1) (password:r3g1), respectively.

The final file tree likes:

```none
dataset
├── WeatherStream
├── ...
code
├── checkpoints
    ├── GT-Rain-Snow
        ├── UNET_model_20.pth
    ├── WeatherStream
        ├── UNET_model_20.pth
├── imperfect-deweathering
    ├── train.py
    ├── test.py
    ├── ...
```

## Inference
Just run this command in ./scripts_eval/eval_unet1.sh:
```bash
echo "Start to test the model...."

name="GT-Rain-Snow"  # or modify to WeatherStream
device="0"  # GPU you used
load_iter=20
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
    --save_imgs True\  # you can modify it to False if you don't want to save images
    --gpu_ids $device\
    -j 4 | tee $LOG
```
## Network Architecture

<img src = "https://i.imgur.com/ILyYCuw.png"> 

## Results
Experiments are conducted with Restormer and RainRobust networks on GT-Rain-Snow and WeatherStream datasets, respectively.

<img src = "https://i.imgur.com/2mheOWr.png"> 

## Visualizations

### Qualitative testing results of the de-weathering models trained with GT-Rain-Snow dataset.
<img src = "https://i.imgur.com/BQfM8Di.png"> 

### Qualitative testing results of the de-weathering models trained with WeatherStream dataset.
<img src = "https://i.imgur.com/BQfM8Di.png"> 

## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{liu2023learning,
  title={Learning Real-World Image De-Weathering with Imperfect Supervision},
  author={Liu, Xiaohui and Zhang, Zhilu and Wu, Xiaohe and Feng, Chaoyu and Wang, Xiaotao and LEI, LEI and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2310.14958},
  year={2023}
}
```

## Acknowledgement

- This repo is built upon the framework if [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and we borrow some code from [GT-Rain](https://github.com/UCLA-VMG/GT-RAIN) and [Restormer](https://github.com/swz30/Restormer), thanks for their excellent work.
