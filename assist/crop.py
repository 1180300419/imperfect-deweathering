'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-10-16 20:11:59
LastEditors: Liu Xiaohui
LastEditTime: 2022-10-19 13:15:08
'''
from logging import root
import os
import cv2

# 将root_path下的图片裁剪成特定大小
root_path = '/home/user/files/data_set/SPA-Dataset/Testing/Real_Internet/'

for img in os.listdir(root_path):
    if img[-3:] == 'png':
        img_png = cv2.imread(os.path.join(root_path, img))
        h, w, c = img_png.shape
        img_png = img_png[:h - h % 4, :w - w % 4, :]
        cv2.imwrite(os.path.join('/home/user/files/data_set/SPA-Dataset/Testing/Real_Internet_crop', img), img_png)
    