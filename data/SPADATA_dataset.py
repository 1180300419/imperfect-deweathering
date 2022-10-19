import re
from . import BaseDataset as BaseDataset
import os
from .imlib import imlib
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool
import random
from util.rain_mask_aug import rain_aug
from util.rotation_data_aug import gen_rotate_image
import torchvision.transforms.functional as TF


class SPADATADataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='SPADATA'):
        super(SPADATADataset, self).__init__(opt, split, dataset_name)

        if self.root == '':
            rootlist = [
                '/hdd1/lxh/derain/dataset/SPA-Dataset',
                '/data/wrh/lxh/derain/dataset/SPA-Dataset',
                '/home/user/files/data_set/SPA-Dataset'
            ]
            for root in rootlist:
                if os.path.isdir(root):
                    self.root = root
                    break
        
        self.patch_size = opt.patch_size
        self.mode = opt.mode  # RGB, Y or L=
        self.imio = imlib(self.mode, lib=opt.imlib)

        self.sigma = 13
        self.zoom_min = .06
        self.zoom_max = 1.8

        # self.rain_mask_dir = os.path.join(self.root, 'Streaks_Garg06')
        self.names, self.scenes, self.rainy_dirs, self.clean_dirs = self._get_image_dir(self.root, split)
        # 还要返回rain_mask dir

        # names 存储rainy img的名字
        if split == 'train':
            self._getitem = self._getitem_trian
        elif split == 'val':
            self._getitem = self._getitem_val
        elif split == 'test':
            self._getitem = self._getitem_test
        else:
            raise ValueError
        
        self.len_data = len(self.names)
        self.rainy_imgs = [0] * len(self.names)
        self.clean_imgs = [0] * len(self.names)

        read_images(self)  # 是不是所有情况直接读取图片，待定

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self._getitem(index)

    def _getitem_trian(self, index):

        # rain mix 
        rainy_img = self.rainy_imgs[index]  # C * H * W
        clean_img = self.clean_imgs[index]
        # rainy_img, clean_img = rain_aug(rainy_img, clean_img, self.rain_mask_dir, zoom_min=self.zoom_min, zoom_max=self.zoom_max)

        # Random rotation
        angle = np.random.normal(0, self.sigma)
        rainy_img_rot = gen_rotate_image(rainy_img, angle)
        if (rainy_img_rot.shape[0] >= 256 and rainy_img_rot.shape[1] >= 256):
            rainy_img = rainy_img_rot
            clean_img = gen_rotate_image(clean_img, angle)

        # reflect pad and random cropping to ensure the right image size for training
        h,w = rainy_img.shape[-2:]
        rainy_img = np.float32(rainy_img / 255.)
        clean_img = np.float32(clean_img / 255.)
       
        # reflect padding
        padw = self.patch_size - w if w < self.patch_size else 0
        padh = self.patch_size - h if h < self.patch_size else 0
        if padw != 0 or padh != 0:
            rainy_img = np.pad(rainy_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')
            clean_img = np.pad(clean_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')

        # random cropping
        rainy_img, clean_img = self._crop_patch(rainy_img, clean_img)
        # random flip
        rainy_img, clean_img = self._clip(rainy_img, clean_img)

        return {
            'rainy_img': rainy_img * 2 - 1,
            'clean_img': clean_img * 2 - 1,
            'file_name': self.names[index]
        }

    def _getitem_val(self, index):
        rainy_img = self.rainy_imgs[index]
        clean_img = self.clean_imgs[index]
        h,w = rainy_img.shape[-2:]

        rainy_img = np.float32(rainy_img / 255.)
        clean_img = np.float32(clean_img / 255.)

        # reflect padding
        padw = self.patch_size - w if w < self.patch_size else 0
        padh = self.patch_size - h if h < self.patch_size else 0
        if padw != 0 or padh != 0:
            rainy_img = np.pad(rainy_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')
            clean_img = np.pad(clean_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')

        rainy_img, clean_img = self._crop_center(rainy_img, clean_img)
        return {
            'rainy_img': rainy_img * 2 - 1,
            'clean_img': clean_img * 2 - 1,
            'file_name': self.names[index]
        }

    def _getitem_test(self, index):
        rainy_img = self.rainy_imgs[index]
        clean_img = self.clean_imgs[index]
        h,w = rainy_img.shape[-2:]

        rainy_img = np.float32(rainy_img / 255.)
        clean_img = np.float32(clean_img / 255.)
        rainy_img = rainy_img[..., :h - h % 4, :w - w % 4]
        clean_img = clean_img[..., :h - h % 4, :w - w % 4]

        return {
            'rainy_img': rainy_img * 2 - 1,
            'clean_img': clean_img * 2 - 1,
            'file_name': self.names[index]
        }   

    def _get_image_dir(self, root, split):
        # 获取所有有雨图片以及对应无雨图片的路径，并且获取所有的有雨图片的名字
        # 返回有雨图片的名字，和与之对应的有雨图片以及无雨图片
        if split == 'train':
            root_path = os.path.join(root, 'Training')
        elif split == 'val':
            root_path = os.path.join(root, 'Testing', 'real_test_1000')
        elif split == 'test':
            root_path = os.path.join(root, 'Testing', 'real_test_1000')
        else:
            raise ValueError
        
        names = []
        rainy_dirs = []
        clean_dirs = []
        # print(os.path.join(root_path, 'rainy'))
        # exit(0)
        for img_path in os.listdir(os.path.join(root_path, 'rainy')):
            names.append(img_path)
            rainy_dirs.append(os.path.join(root_path, 'rainy', img_path))
            clean_dirs.append(os.path.join(root_path, 'gt', img_path))

        return names, rainy_dirs, clean_dirs

    def _crop_patch(self, rainy_img, clean_img):
        hh, ww = rainy_img.shape[-2:]
        pw = random.randrange(0, ww - self.patch_size + 1)
        ph = random.randrange(0, hh - self.patch_size + 1)
        
        rainy_img = rainy_img[..., ph:ph+self.patch_size, pw:pw+self.patch_size]
        clean_img = clean_img[..., ph:ph+self.patch_size, pw:pw+self.patch_size]
        return rainy_img, clean_img
                
    def _crop_center(self, rainy_img, clean_img, p=256):
        hh, ww = rainy_img.shape[-2:]
        begin_h, begin_w = hh // 2 - self.patch_size // 2, ww // 2 - self.patch_size // 2
        rainy_img = rainy_img[..., begin_h:begin_h+self.patch_size, begin_w:begin_w+self.patch_size]
        clean_img = clean_img[..., begin_h:begin_h+self.patch_size, begin_w:begin_w+self.patch_size]
        return rainy_img, clean_img

    def _clip(self, rainy_img, clean_img):
        aug = random.randint(0, 2)
        if aug == 1:
            rainy_img, clean_img = rainy_img[..., ::-1], clean_img[..., ::-1]
        elif aug == 2:
            rainy_img, clean_img = rainy_img[..., ::-1, :], clean_img[..., ::-1, :]
        return np.ascontiguousarray(rainy_img), np.ascontiguousarray(clean_img)

def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
	# Due to the memory (32 GB) limitation, here we only preload the raw images. 
	# If you have enough memory, you can also modify the code to preload the sRGB images to speed up the training process.
    i, obj = arg
    for _ in range(3):
        try:
            obj.rainy_imgs[i] = obj.imio.read(obj.rainy_dirs[i])
            obj.clean_imgs[i] = obj.imio.read(obj.clean_dirs[i])
            failed = False
            break
        except:
            failed = True
    if failed: print(i, '%s fails!' % obj.names[i])

def read_images(obj):
	# may use `from multiprocessing import Pool` instead, but less efficient and
	# NOTE: `multiprocessing.Pool` will duplicate given object for each process.
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
		pass
	pool.close()
	pool.join()              