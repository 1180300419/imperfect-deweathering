'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-09-16 12:39:08
LastEditors: Liu Xiaohui
LastEditTime: 2022-11-15 19:47:13
'''
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from util.util import calc_lpips 
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
import struct
import cv2
import lpips


if __name__ == '__main__':
    opt = TestOptions().parse()
    loss_fn_alex_1 = lpips.LPIPS(net='alex', version='0.1')
    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset, ncols=85)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        
        model = create_model(opt)
        model.setup(opt)
        model.eval()
       
        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            tqdm_val.reset()
            psnr_dict = dict()
            ssim_dict = dict()
            lpips_dict = dict()
            psnr = [0.0] * dataset_size_test
            ssim = [0.0] * dataset_size_test
            lpipses = [0.0] * dataset_size_test
            time_val = 0
            count = 0
            
            for i, data in enumerate(tqdm_val):
                model.set_input(data, load_iter)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()
                
                test_scenes = os.listdir('../../dataset/GT-Rain/GT-RAIN_test')

                if opt.calc_metrics:
                    b = res['clean_img'].shape[0]
                    for tmp_b in range(b):
                        lpipses[count] = calc_lpips(res['clean_img'], res['derained_img'], loss_fn_alex_1, 'cuda:' + str(opt.gpu_ids[0]))
                        derained = np.array(res['derained_img'][tmp_b].cpu()).astype(np.uint8).transpose((1, 2, 0)) / 255.
                        clean = np.array(res['clean_img'][tmp_b].cpu()).astype(np.uint8).transpose((1, 2, 0)) / 255.
                        psnr[count] = calc_psnr(clean, derained, data_range=1.)
                        ssim[count] = calc_ssim(clean, derained, multichannel=True)
                        if data['folder'][tmp_b] in psnr_dict:
                            psnr_dict[data['folder'][tmp_b]].append(psnr[count])
                            ssim_dict[data['folder'][tmp_b]].append(ssim[count])
                            lpips_dict[data['folder'][tmp_b]].append(lpipses[count])
                        else:
                            psnr_dict[data['folder'][tmp_b]] = [psnr[count]]
                            ssim_dict[data['folder'][tmp_b]] = [ssim[count]]
                            lpips_dict[data['folder'][tmp_b]] = [lpipses[count]]
                        count += 1
                    
                if opt.save_imgs:
                    save_dir_rgb = os.path.join('../checkpoints', opt.name, 'test_epoch_' + str(opt.load_iter), 'rgb_out', data['folder'][0])
                    os.makedirs(save_dir_rgb, exist_ok=True)
                    out_img = np.array(res['derained_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'd.png'), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                    rainy_img = np.array(res['single_rainy_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'r.png'), cv2.cvtColor(rainy_img, cv2.COLOR_RGB2BGR))
                    clean_img = np.array(res['clean_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'c.png'), cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))

    for dataset in datasets:
        datasets[dataset].close()


# test_scenes = os.listdir('/hdd1/lxh/derain/dataset/GT-Rain/GT-RAIN_test')
test_scenes = os.listdir('../../dataset/GT-Rain/GT-RAIN_test')

rain_psnr = []
rain_ssim = []
rain_lpips = []

fog_psnr = []
fog_ssim = []
fog_lpips = []

snow_psnr = []
snow_ssim = []
snow_lpips = []

for key in sorted(psnr_dict.keys()):
    if key in test_scenes or 'rain' in key:
        rain_psnr += psnr_dict[key]
        rain_ssim += ssim_dict[key]
        rain_lpips += lpips_dict[key]
    elif 'fog' in key:
        fog_psnr += psnr_dict[key]
        fog_ssim += ssim_dict[key]
        fog_lpips += lpips_dict[key]
    elif 'snow' in key:
        snow_psnr += psnr_dict[key]
        snow_ssim += ssim_dict[key]
        snow_lpips += lpips_dict[key]
        
avg_rain_psnr_rgb = '%.2f'%np.mean(rain_psnr)
avg_rain_ssim_rgb = '%.4f'%np.mean(rain_ssim)
avg_rain_lpips_rgb = '%.4f'%np.mean(rain_lpips)
print('\nrain\ntotal: PSNR: %s SSIM: %s LPIPS: %s' % (avg_rain_psnr_rgb, avg_rain_ssim_rgb, avg_rain_lpips_rgb))

avg_fog_psnr_rgb = '%.2f'%np.mean(fog_psnr)
avg_fog_ssim_rgb = '%.4f'%np.mean(fog_ssim)
avg_fog_lpips_rgb = '%.4f'%np.mean(fog_lpips)
print('\nfog\ntotal: PSNR: %s SSIM: %s LPIPS: %s' % (avg_fog_psnr_rgb, avg_fog_ssim_rgb, avg_fog_lpips_rgb))

avg_snow_psnr_rgb = '%.2f'%np.mean(snow_psnr)
avg_snow_ssim_rgb = '%.4f'%np.mean(snow_ssim)
avg_snow_lpips_rgb = '%.4f'%np.mean(snow_lpips)
print('\nsnow\ntotal: PSNR: %s SSIM: %s LPIPS: %s' % (avg_snow_psnr_rgb, avg_snow_ssim_rgb, avg_snow_lpips_rgb))

avg_psnr_rgb = '%.2f'%np.mean(psnr)
avg_ssim_rgb = '%.4f'%np.mean(ssim)
avg_lpips_rgb = '%.4f'%np.mean(lpipses)
print('\nEpoch %d \ntotal: PSNR: %s SSIM: %s LPIPS: %s' % (opt.load_iter, avg_psnr_rgb, avg_ssim_rgb, avg_lpips_rgb))
