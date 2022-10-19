'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-09-16 12:39:08
LastEditors: Liu Xiaohui
LastEditTime: 2022-10-16 17:32:07
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
from skimage import exposure


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
        print(len(dataset))
        datasets[dataset_name] = tqdm(dataset)

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

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            psnr = [0.0] * dataset_size_test
            ssim = [0.0] * dataset_size_test
            lpipses = [0.0] * dataset_size_test
            time_val = 0
            # print(dataset_size_test)
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()
                
                if opt.calc_metrics:
                    
                    derained = np.array(res['derained_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0)) / 255.
                    clean = np.array(res['clean_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0)) / 255.

                    derained = exposure.match_histograms(derained, clean, multichannel=True)
                    derained_ten = torch.from_numpy(derained.transpose(2, 0, 1)).unsqueeze(0).to(opt.gpu_ids[0])
                    clean_histed = exposure.match_histograms(clean, derained, multichannel=True)
                    # print(res['clean_img'].shape, derained_ten.shape)
                    psnr[i] = calc_psnr(clean, derained, data_range=1.)
                    ssim[i] = calc_ssim(clean, derained, multichannel=True)
                    lpipses[i] = calc_lpips(res['clean_img'], derained_ten.float(), loss_fn_alex_1, 'cuda:' + str(opt.gpu_ids[0]))
                    
                    
                if opt.save_imgs:
                    save_dir_rgb = os.path.join('../checkpoints', opt.name, 'test_epoch_' + str(opt.load_iter), 'rgb_out', data['file_name'][0][:-17])
                    os.makedirs(save_dir_rgb, exist_ok=True)

                    out_img = np.array(res['derained_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'd.png'), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                    rainy_img = np.array(res['rainy_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'r.png'), cv2.cvtColor(rainy_img, cv2.COLOR_RGB2BGR))
                    clean_img = np.array(res['clean_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'c.png'), cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
                    # print(derained.shape)
                    derained = (derained * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'h.png'), cv2.cvtColor(derained, cv2.COLOR_RGB2BGR))
                    clean_histed = (clean_histed * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'ch.png'), cv2.cvtColor(clean_histed, cv2.COLOR_RGB2BGR))
            avg_psnr_rgb = '%.2f'%np.mean(psnr)
            avg_ssim_rgb = '%.4f'%np.mean(ssim)
            avg_lpips_rgb = '%.4f'%np.mean(lpipses)

            print('Epoch %d Time: %.3f s AVG Time: %.3f ms PSNR: %s SSIM: %s LPIPS: %s\n' % (opt.load_iter, time_val, time_val/dataset_size_test*1000, avg_psnr_rgb, avg_ssim_rgb, avg_lpips_rgb))

    for dataset in datasets:
        datasets[dataset].close()

