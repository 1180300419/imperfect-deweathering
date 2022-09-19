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


class GTRAINDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='GT-Rain'):
        super(GTRAINDataset, self).__init__(opt, split, dataset_name)

        if self.root == '':
            rootlist = [
                '/hdd1/lxh/derain/dataset/GT-Rain/',
                '/data/wrh/lxh/derain/dataset/GT-Rain',
                '/home/user/files/data_set/GT-Rain'
            ]
            for root in rootlist:
                if os.path.isdir(root):
                    self.root = root
                    break
        
        self.patch_size = opt.patch_size
        self.mode = opt.mode  # RGB, Y or L=
        self.imio = imlib(self.mode, lib=opt.imlib)

        self.names, self.rainy_dirs = self._get_image_dir(self.root, split)

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

    def _getitem_test(self, index):
        rainy_img = self.rainy_imgs[index]
        h,w = rainy_img.shape[-2:]

        rainy_img = np.float32(rainy_img / 255.)
        rainy_img = rainy_img[..., :h - h % 4, :w - w % 4]

        return {
            'rainy_img': rainy_img * 2 - 1,
            'file_name': self.names[index]
        }   

    def _get_image_dir(self, root, split):
        root_path = os.path.join(root, 'Real_Internet')
        
        

        names = []
        rainy_dirs = []

        for img in os.listdir(root_path):
            rainy_dirs.append(os.path.join(root_path, img))

        return names, rainy_dirs

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