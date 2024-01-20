"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time
from functools import wraps
import torch
import random
import numpy as np
import cv2
import torch
# import colour_demosaicing
import glob
import lpips
import torch.nn.functional as F
from math import exp
from skimage.metrics import structural_similarity as ssim

# 修饰函数，重新尝试600次，每次间隔1秒钟
# 能对func本身处理，缺点在于无法查看func本身的提示
def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(600):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                time.sleep(1)
        return ret
    return wrapper

# 修改后的print函数及torch.save函数示例
@loop_until_success
def loop_print(*args, **kwargs):
    print(*args, **kwargs)

@loop_until_success
def torch_save(*args, **kwargs):
    torch.save(*args, **kwargs)

def calc_psnr(sr, hr, range=1.0):
    # shave = 2
    with torch.no_grad():
        diff = (sr - hr) / range
        # diff = diff[:, :, shave:-shave, shave:-shave]
        mse = torch.pow(diff, 2).mean()
        # print(mse)
        return (-10 * torch.log10(mse)).item()
 
def calc_ssim(sr, hr):
    sr = sr[0].data.cpu().numpy()
    hr = hr[0].data.cpu().numpy()
    # print(sr.shape)
    return ssim(sr, hr, channel_axis=0, win_size=11, data_range=1.0, gaussian_weights=True)



def calc_lpips(sr, hr, loss_fn_alex_1, device):
    sr = sr / (255. / 2.) - 1
    hr = hr / (255. / 2.) - 1
    sr = sr.to(device)
    hr = hr.to(device)
    loss_fn_alex_1 = loss_fn_alex_1.to(device)
    LPIPS_1 = loss_fn_alex_1(sr, hr)
    return LPIPS_1.detach().cpu().data.numpy() #, LPIPS_1.detach().cpu()

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count

def print_numpy(x, val=True, shp=True):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, mid = %3.3f, std=%3.3f'
              % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def prompt(s, width=66):
    print('='*(width+4))
    ss = s.split('\n')
    if len(ss) == 1 and len(s) <= width:
        print('= ' + s.center(width) + ' =')
    else:
        for s in ss:
            for i in split_str(s, width):
                print('= ' + i.ljust(width) + ' =')
    print('='*(width+4))

def split_str(s, width):
    ss = []
    while len(s) > width:
        idx = s.rfind(' ', 0, width+1)
        if idx > width >> 1:
            ss.append(s[:idx])
            s = s[idx+1:]
        else:
            ss.append(s[:width])
            s = s[width:]
    if s.strip() != '':
        ss.append(s)
    return ss

def augment_func(img, hflip, vflip, rot90):  # CxHxW
    if hflip:   img = img[:, :, ::-1]
    if vflip:   img = img[:, ::-1, :]
    if rot90:   img = img.transpose(0, 2, 1)
    return np.ascontiguousarray(img)

def augment(*imgs):  # CxHxW
    # no need 
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    # no need
    rot90 = random.random() < 0.5
    return (augment_func(img, hflip, vflip, rot90) for img in imgs)

def remove_black_level(img, black_lv=63, white_lv=4*255):
    img = np.maximum(img.astype(np.float32)-black_lv, 0) / (white_lv-black_lv)
    return img

def gamma_correction(img, r=1/2.2):
    img = np.maximum(img, 0)
    img = np.power(img, r)
    return img

def extract_bayer_channels(raw):  # HxW
    ch_R  = raw[0::2, 0::2]
    ch_Gb = raw[0::2, 1::2]
    ch_Gr = raw[1::2, 0::2]
    ch_B  = raw[1::2, 1::2]
    raw_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    raw_combined = np.ascontiguousarray(raw_combined.transpose((2, 0, 1)))
    return raw_combined  # 4xHxW

def get_coord(H, W, x=448/3968, y=448/2976):
    x_coord = np.linspace(-x + (x / W), x - (x / W), W)
    x_coord = np.expand_dims(x_coord, axis=0)
    x_coord = np.tile(x_coord, (H, 1))
    x_coord = np.expand_dims(x_coord, axis=0)

    y_coord = np.linspace(-y + (y / H), y - (y / H), H)
    y_coord = np.expand_dims(y_coord, axis=1)
    y_coord = np.tile(y_coord, (1, W))
    y_coord = np.expand_dims(y_coord, axis=0)

    coord = np.ascontiguousarray(np.concatenate([x_coord, y_coord]))
    coord = np.float32(coord)

    return coord

def read_wb(txtfile, key):
    wb = np.zeros((1,4))
    with open(txtfile) as f:
        for l in f:
            if key in l:
                for i in range(wb.shape[0]):
                    nextline = next(f)
                    try:
                        wb[i,:] = nextline.split()
                    except:
                        print("WB error XXXXXXX")
                        print(txtfile)
    wb = wb.astype(np.float32)
    return wb

def normalization(data):
    _range = np.max(data) - np.min(data)
    # print(np.max(data) , np.min(data))
    return (data - np.min(data)) / _range

def FFTfusion(leftLR_img, rightLR_img):
    fft_leftLR = np.fft.fft2(leftLR_img)
    fft_rightLR = np.fft.fft2(rightLR_img)
    fft_leftLR_r = np.zeros(leftLR_img.shape, dtype = complex)
    fft_leftLR_r.real = np.abs(fft_rightLR) * np.cos(np.angle(fft_leftLR))
    fft_leftLR_r.imag = np.abs(fft_rightLR) * np.sin(np.angle(fft_leftLR))
    fft_rightLR_l = np.zeros(rightLR_img.shape, dtype = complex)
    fft_rightLR_l.real = np.abs(fft_leftLR) * np.cos(np.angle(fft_rightLR))
    fft_rightLR_l.imag = np.abs(fft_leftLR) * np.sin(np.angle(fft_rightLR))
    leftLR_r = normalization(np.fft.ifft2(fft_leftLR_r))
    rightLR_l = normalization(np.fft.ifft2(fft_rightLR_l))
    return np.float32(np.abs(leftLR_r)), np.float32(np.abs(rightLR_l))


def rgb2ycbcr(img, only_y=True):
    """
    same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def rgbten2ycbcrten(img_ten, only_y=True):
    """
    img_ten: torch.Tensor, [-1, 1]
    """
    img_ten = torch.clamp((img_ten * 0.5 + 0.5) * 255, 0, 255).round()
    img_ten = img_ten.permute(0, 2, 3, 1)
    # convert
    if only_y:
        coef = torch.tensor([65.481, 128.553, 24.966], device=img_ten.device)
        rlt = torch.matmul(img_ten, coef) / 255.0 + 16.0
        rlt = rlt / (255. / 2) - 1.
        b, h, w, c = rlt.shape
        rlt = rlt.reshape((b, c, h, w))
        return rlt
    else:
        coef = torch.tensor( [
                            [65.481, -37.797, 112.0], 
                            [128.553, -74.203, -93.786],
                            [24.966, 112.0, -18.214]], device=img_ten.device)
        rlt = torch.matmul(img_ten, coef) / 255.0 + torch.tensor([16, 128, 128], device=img_ten.device)
        rlt = rlt / (255. / 2) - 1.
        rlt = rlt.permute(0, 3, 1, 2)
        return rlt

def bgr2ycbcr(img, only_y=True):
    """
    bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
    
def bgrten2ycbcrten(img_ten, only_y=True):
    pass

def rgb2gray(rgb_img):
    rgb_img = rgb_img.transpose((1, 2, 0))
    rgb_img = np.dot(rgb_img, [0.299, 0.587, 0.114])
    h, w = rgb_img.shape
    rgb_img_ret = rgb_img.reshape((-1, h, w))
    return rgb_img_ret

def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1.,  4.,  6.,  4., 1.],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1.,  4.,  6.,  4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def laplacian_pyramid(img, kernel, max_levels=3):
    assert max_levels > 1
    current = img
    pyr = []
    for level in range(max_levels - 1):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(down)
    return pyr

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    x_up = torch.zeros(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2, device=x.device)
    x_up[:, :, ::2, ::2] = x
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def crop_random(img_ten1, img_ten2, patch_size=224):
    hh, ww = img_ten1.shape[-2:]
    pw = random.randrange(0, ww - patch_size + 1)
    ph = random.randrange(0, hh - patch_size + 1)
    
    img_ten1 = img_ten1[..., ph : ph + patch_size, pw : pw + patch_size]
    img_ten2 = img_ten2[..., ph : ph + patch_size, pw : pw + patch_size]
    return img_ten1, img_ten2
            
def crop_center(img_ten1, img_ten2, patch_size=224):
    hh, ww = img_ten1.shape[-2:]
    begin_h, begin_w = hh // 2 - patch_size // 2, ww // 2 - patch_size // 2
    img_ten1 = img_ten1[..., begin_h : begin_h + patch_size, begin_w : begin_w + patch_size]
    img_ten2 = img_ten2[..., begin_h : begin_h + patch_size, begin_w : begin_w + patch_size]
    return img_ten1, img_ten2