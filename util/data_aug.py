'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-11-01 15:44:40
LastEditors: Liu Xiaohui
LastEditTime: 2022-11-06 16:50:33
'''
import numpy as np
import random


def cutblur(im1, im2, prob=1.0, alpha=1.0):
    # im1: clean img
    # im2: rainy_img
    if im1.shape != im2.shape:
        raise ValueError('im1 and im2 have to be the same resolution')
    
    if alpha <= 0 or np.random.rand(1) > prob:
        return im2
    
    cut_ratio = np.random.randn() * 0.01 + alpha

    _, h, w = im2.shape
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)

    cy = np.random.randint(0, h - ch + 1)
    cx = np.random.randint(0, w - cw + 1)

    # if np.random.random() > 0.5:
    im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    # else:
    #     im2_aug = im1.copy()
    #     im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
    #     im2 = im2_aug
    return im2

def cutblur_patch(im1, im2, prob=1.0, alpha=1.0, alpha_patch=1.0):
    # im1: clean img
    # im2: rainy_img
    # prob:使用cutblur的概率
    # alpha：使用cutblur的边长比
    # alpha_patch: 每个小patch边长比
    _, h, w = im2.shape
    # assert (alpha * h) % (alpha_patch * h) == 0, 'alpha and alpha_path mismatch'
    if im1.shape != im2.shape:
        raise ValueError('im1 and im2 have to be the same resolution')
    
    if alpha <= 0 or np.random.rand(1) > prob:
        return im2
    
   
    patch_h, patch_w = int(h * alpha_patch), int(w * alpha_patch)

    patch_num = int(alpha // alpha_patch)
    
    cy = random.sample(range(0, int(1 // alpha_patch)), patch_num)
    cx = random.sample(range(0, int(1 // alpha_patch)), patch_num)
    if np.random.random() > 0.5:
        for i in range(patch_num):
            begin_y = cy[i] * patch_h
            begin_x = cx[i] * patch_w
            im2[..., begin_y:begin_y+patch_h, begin_x:begin_x+patch_w] = im1[..., begin_y:begin_y+patch_h, begin_x:begin_x+patch_w]
    else:
        img2_aug = im1.copy()
        for i in range(patch_num):
            begin_y = cy[i] * patch_h
            begin_x = cx[i] * patch_w
            img2_aug[..., begin_y:begin_y+patch_h, begin_x:begin_x+patch_w] = im2[..., begin_y:begin_y+patch_h, begin_x:begin_x+patch_w]
        im2 = img2_aug
        
    return im2


def random_mask(input_img, patch_size, mask_per):
    # input_img: 需要进行mask操作的图片，大小是C*H*W
    # patch_size: mask的小patch大小
    # mask_per: mask掉的小patch所占比例
    C, H, W = input_img.shape
    assert H % patch_size == 0 and W % patch_size == 0, "wrong patch_size"
    assert mask_per >= 0 and mask_per <= 1, "wrong mask percent"
    
    col_num = int(W // patch_size)
    row_num = int(H // patch_size)
    tol_mask_num = col_num * row_num  # 小patch总数
    mask_num = int(tol_mask_num * mask_per)  #　要进行mask的小patch总数
    mask_index = random.sample(range(0, tol_mask_num + 1), mask_num)
    
    for index in mask_index:
        row = index // col_num
        col = index - row * col_num - 1
        
        begin_row, begin_col = row * patch_size, col * patch_size
        input_img[begin_row:begin_row+patch_size, begin_col:begin_col+patch_size] = 0
    return input_img
        
    