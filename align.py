'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-10-19 13:19:38
LastEditors: Liu Xiaohui
LastEditTime: 2022-10-19 16:29:00
'''
from logging import root
import os
import cv2
from cv2 import mean
import torch
import numpy as np
from alignment.pwcnet import PWCNet, backwarp

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from util.util import calc_lpips
from lpips import lpips

# 将指定目录下的图片（Rainy、Derained和GT进行对齐处理，查看对齐效果以及对齐之后的各项指标）
rainy_root = '/home/user/files/data_set/GT-Rain/GT-RAIN_val'
device = 'cuda:0'
loss_fn_alex_1 = lpips.LPIPS(net='alex', version='0.1')
dst_path = '/home/user/code/derain/checkpoints/aligned_rainy_img'
align_net = PWCNet(load_pretrained=True, weights_path='../ckpt/pwcnet-network-default.pth')
align_net = align_net.to(device)
for param in align_net.parameters():
    param.requires_grad = False
# align_net.eval()

unaligned_psnr = []
psnr = []
ssim = []
lpipses = []

for scene in os.listdir(rainy_root):
    for img_name in os.listdir(os.path.join(rainy_root, scene)):
        if img_name[-9] != 'R':
            continue
        if scene == 'Gurutto_1-2':
            rainy_img_name = os.path.join(rainy_root, scene, img_name)
            clean_img_name = os.path.join(rainy_root, scene, img_name[:-9] + 'C' + img_name[-8:])
        else:
            rainy_img_name = os.path.join(rainy_root, scene, img_name)
            clean_img_name = os.path.join(rainy_root, scene, img_name[:-9] + 'C-000.png')
        
        rainy_img = cv2.imread(rainy_img_name)
        rainy_img_ten = rainy_img[..., ::-1].transpose((2, 0, 1))
        clean_img = cv2.imread(clean_img_name)
        clean_img_ten = clean_img[..., ::-1].transpose((2, 0, 1))
        
        rainy_img_ten = torch.from_numpy(rainy_img_ten.copy()).to(device)
        clean_img_ten = torch.from_numpy(clean_img_ten.copy()).to(device)
        # rainy_img_ten = rainy_img_ten[..., ::-1]
        # clean_img_ten = clean_img_ten[..., ::-1]
        rainy_img_ten = rainy_img_ten / 255.
        clean_img_ten = clean_img_ten / 255.
        rainy_img_ten = rainy_img_ten.unsqueeze(0)
        clean_img_ten = clean_img_ten.unsqueeze(0)
        with torch.no_grad():
            align_net.eval()
            flow = align_net(rainy_img_ten, clean_img_ten)
            aligned_rainy_img_ten = backwarp(rainy_img_ten, flow)
            
            lpipses.append(0)
            # calc_lpips(clean_img_ten, rainy_img_ten, loss_fn_alex_1, device)
            aligned_rainy_img = aligned_rainy_img_ten.detach()[0].cpu().numpy()  # 转换成numpy并且转换到CPU上
            aligned_rainy_img = aligned_rainy_img[::-1, ...]  # RGB转换成BGR
            aligned_rainy_img = aligned_rainy_img.transpose((1, 2, 0))
            aligned_rainy_img = (aligned_rainy_img * 255).astype(np.uint8)
            unaligned_psnr.append(calc_psnr(clean_img[8:-8, 8:-8, :], rainy_img[8:-8, 8:-8, :], data_range=255))
            psnr.append(calc_psnr(clean_img[8:-8, 8:-8, :], aligned_rainy_img[8:-8, 8:-8, :], data_range=255))
            ssim.append(0)
            # calc_ssim(clean_img, aligned_rainy_img, multichannel=True)
            os.makedirs(os.path.join(dst_path, scene), exist_ok=True)
            print(os.path.join(dst_path, scene))
            cv2.imwrite(os.path.join(dst_path, scene, img_name), aligned_rainy_img)
            print('done')

print('unaligned: ', np.mean(unaligned_psnr))
print(np.mean(psnr), np.mean(ssim), np.mean(lpipses))