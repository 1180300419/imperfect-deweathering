import re
import scipy as sp
from . import BaseDataset as BaseDataset
import os
from .imlib import imlib
import numpy as np
from tqdm import tqdm
from multiprocessing.dummy import Pool
import random
from util.rain_mask_aug import rain_aug_list
from util.snow_mask_aug import snow_aug_list
from util.rotation_data_aug import gen_rotate_image
import torchvision.transforms.functional as TF
from util.data_aug import cutblur, random_mask
from natsort import natsorted
from glob import glob
import cv2

class MULGTWEADataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='GT-Rain'):
        super(MULGTWEADataset, self).__init__(opt, split, dataset_name)

        if self.root == '':
            rootlist = [
                '/hdd1/lxh/derain/dataset/GT-Rain/',
                '/data/wrh/lxh/derain/dataset/GT-Rain',
                '/home/user/files/data_set/GT-Rain',
                '/mnt/disk10T/lxh/derain/dataset/GT-Rain'
            ]
            for root in rootlist:
                if os.path.isdir(root):
                    self.root = root
                    break
        self.patch_size = opt.patch_size
        self.data_section = opt.data_section
        self.mode = opt.mode  # RGB, Y or L=
        self.imio = imlib(self.mode, lib=opt.imlib)

        self.length = opt.input_frames
        self.sigma = 13
        self.zoom_min = .06
        self.zoom_max = 1.8

        self.rain_mask_dir = os.path.join(self.root, 'Streaks_Garg06')
        self.names, self.scenes, self.rainy_dirs, self.clean_dirs, self.scene_indices, self.wgt_or_not = self._get_image_dir(self.root, split)
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
        # self.rainy_imgs = [0] * len(self.names)
        # self.clean_imgs = [0] * len(self.names)

        # read_images(self)  # 是不是所有情况直接读取图片，待定

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self._getitem(index)
    
    def _getitem_trian(self, index):
        
        for i in range(len(self.scene_indices)):
            if index >= self.scene_indices[i][0] and index <= self.scene_indices[i][-1]:
                scene_index = i
                break
        
        half_length = self.length // 2
        
        if index + half_length <= self.scene_indices[scene_index][-1]:
            begin_index = max(index - half_length, self.scene_indices[scene_index][0])
        else:
            begin_index = self.scene_indices[scene_index][-1] - (self.length - 1)
        
        indexes = []
        for i in range(self.length):
            indexes.append(begin_index + i)

        input_rainy_img_list = []
        input_clean_img_list = []
        names = []
        rainy_img_list = []
        clean_img_list = []
        
        count = 0
        for tmp_index in indexes:   
            failed = True
            try:
                input_rainy_img_list.append(self.imio.read(self.rainy_dirs[tmp_index]))
                input_clean_img_list.append(self.imio.read(self.clean_dirs[tmp_index]))
                names.append(self.names[tmp_index])
                failed = False
            except:
                failed = True
            if failed: print(tmp_index, '%s fails' % self.names[tmp_index])

        wgt = self.wgt_or_not[indexes[0]]
        if 'GT-RAIN_train' in self.rainy_dirs[index]: 
            input_rainy_img_list, input_clean_img_list = rain_aug_list(input_rainy_img_list, input_clean_img_list, self.rain_mask_dir, zoom_min=self.zoom_min, zoom_max=self.zoom_max, length=self.length)
        elif 'GT-SNOW_train' in self.rainy_dirs[index]:
            input_rainy_img_list = snow_aug_list(input_rainy_img_list)

        for i in range(self.length):
            rainy_img = input_rainy_img_list[i]
            clean_img = input_clean_img_list[i]
            # Random rotation
            if count == 0:
                angle = np.random.normal(0, self.sigma)
            rainy_img_rot = gen_rotate_image(rainy_img, angle)
            if (rainy_img_rot.shape[0] >= 256 and rainy_img_rot.shape[1] >= 256):
                rainy_img = rainy_img_rot
                clean_img = gen_rotate_image(clean_img, angle)

            # reflect pad and random cropping to ensure the right image size for training
            if count == 0:
                h,w = rainy_img.shape[-2:]
                padw = self.patch_size - w if w < self.patch_size else 0
                padh = self.patch_size - h if h < self.patch_size else 0
                
            if padw != 0 or padh != 0:
                rainy_img = np.pad(rainy_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')
                clean_img = np.pad(clean_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')
        
            # random cropping
            if count == 0:
                hh, ww = rainy_img.shape[-2:]
                pw = random.randrange(0, ww - self.patch_size + 1)
                ph = random.randrange(0, hh - self.patch_size + 1)
                
            rainy_img, clean_img = self._crop_patch(rainy_img, clean_img, ph, pw)
            # random flip
            if count == 0:
                aug = random.randint(0, 2)
            rainy_img, clean_img = self._flip(rainy_img, clean_img, aug)

            h,w = rainy_img.shape[-2:]
            rainy_img = np.float32(rainy_img / 255.)
            clean_img = np.float32(clean_img / 255.)
            
            if self.data_section == '-1-1':
                rainy_img = rainy_img * 2 - 1
                clean_img = clean_img * 2 - 1
            count += 1
                
            rainy_img_list.append(rainy_img)
            clean_img_list.append(clean_img)
            
        return {
            'rainy_img': rainy_img_list,
            'single_rainy_img': rainy_img_list[self.length // 2],
            'clean_img': clean_img_list[self.length // 2],
            'file_name': names[self.length // 2],
            'wgt': wgt
        }

    def _getitem_val(self, index):
        for i in range(len(self.scene_indices)):
            if index >= self.scene_indices[i][0] and index <= self.scene_indices[i][-1]:
                scene_index = i
                break
        
        half_length = self.length // 2
        
        if index + half_length <= self.scene_indices[scene_index][-1]:
            begin_index = max(index - half_length, self.scene_indices[scene_index][0])
        else:
            begin_index = self.scene_indices[scene_index][-1] - (self.length - 1)
        
        indexes = []
        for i in range(self.length):
            indexes.append(begin_index + i)
        
        input_rainy_img_list = []
        input_clean_img_list = []
        names = []
        rainy_img_list = []
        clean_img_list = []

        for tmp_index in indexes:   
            failed = True
            try:
                input_rainy_img_list.append(self.imio.read(self.rainy_dirs[tmp_index]))
                input_clean_img_list.append(self.imio.read(self.clean_dirs[tmp_index]))
                names.append(self.names[tmp_index])
                failed = False
            except:
                failed = True
            if failed: print(tmp_index, '%s fails' % self.names[tmp_index])
        
        wgt = self.wgt_or_not[indexes[0]]

        for i in range(self.length):
            rainy_img = input_rainy_img_list[i]
            clean_img = input_clean_img_list[i]
            
            rainy_img = np.float32(rainy_img / 255.)
            clean_img = np.float32(clean_img / 255.)
            h, w = rainy_img.shape[-2:]
            # reflect padding
            padw = self.patch_size - w if w < self.patch_size else 0
            padh = self.patch_size - h if h < self.patch_size else 0
            if padw != 0 or padh != 0:
                rainy_img = np.pad(rainy_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')
                clean_img = np.pad(clean_img, ((0, 0), (0, padh), (0, padw)), mode='reflect')

            rainy_img, clean_img = self._crop_center(rainy_img, clean_img)
            
            if self.data_section == '-1-1':
                rainy_img = rainy_img * 2 - 1
                clean_img = clean_img * 2 - 1
                
            rainy_img_list.append(rainy_img)
            clean_img_list.append(clean_img)
            
        return {
            'rainy_img': rainy_img_list,
            'single_rainy_img': rainy_img_list[self.length // 2],
            'clean_img': clean_img_list[self.length // 2],
            'file_name': names[self.length // 2],
            'wgt': wgt
        }

    def _getitem_test(self, index):
        
        for i in range(len(self.scene_indices)):
            if index >= self.scene_indices[i][0] and index <= self.scene_indices[i][-1]:
                scene_index = i
                break
            
        half_length = self.length // 2
        
        if index + half_length <= self.scene_indices[scene_index][-1]:
            begin_index = max(index - half_length, self.scene_indices[scene_index][0])
        else:
            begin_index = self.scene_indices[scene_index][-1] - (self.length - 1)
        
        indexes = []
        for i in range(self.length):
            indexes.append(begin_index + i)
            
        rainy_img_list = []
        clean_img_list = []
        names = []

        for tmp_index in indexes:   
            failed = True
            try:
                rainy_img = self.imio.read(self.rainy_dirs[tmp_index])
                clean_img = self.imio.read(self.clean_dirs[tmp_index])
                failed = False
            except:
                failed = True
            if failed: print(tmp_index, '%s fails' % self.names[tmp_index])

            h,w = rainy_img.shape[-2:]

            rainy_img = np.float32(rainy_img / 255.)
            clean_img = np.float32(clean_img / 255.)
            rainy_img = rainy_img[..., :h - h % 4, :w - w % 4]
            clean_img = clean_img[..., :h - h % 4, :w - w % 4]

            if self.data_section == '-1-1':
                rainy_img = rainy_img * 2 - 1
                clean_img = clean_img * 2 - 1

            rainy_img_list.append(rainy_img)
            clean_img_list.append(clean_img)
            names.append(self.names[tmp_index])
            
        return {
            'rainy_img': rainy_img_list,
            'single_rainy_img': rainy_img_list[self.length // 2],
            'clean_img': clean_img_list[self.length // 2],
            'file_name': self.names[index],
            'folder': os.path.basename(os.path.normpath(os.path.dirname(self.rainy_dirs[index])))
        }

    def _get_image_dir(self, root, split):
        # 获取所有有雨图片以及对应无雨图片的路径，并且获取所有的有雨图片的名字
        # 返回有雨图片的名字，和与之对应的有雨图片以及无雨图片
        root_paths = []
        if split == 'train':
            root_paths.append(os.path.join(root, 'GT-RAIN_train'))
            root_paths.append(os.path.join(root, 'GT-SNOW_train'))
            # root_paths.append(os.path.join('/hdd1/lxh/derain/dataset/WeatherStream', 'WeatherStream_train/WeatherStream_train_rain'))
            # root_paths.append(os.path.join('/hdd1/lxh/derain/dataset/WeatherStream', 'WeatherStream_train/WeatherStream_train_snow'))
        elif split == 'val':
            root_paths.append(os.path.join('/hdd1/lxh/derain/dataset/WeatherStream', 'WeatherStream_test'))
            root_paths.append(os.path.join(root, 'GT-RAIN_test'))
        elif split == 'test':
            root_paths.append(os.path.join('/hdd1/lxh/derain/dataset/WeatherStream', 'WeatherStream_test'))
            root_paths.append(os.path.join(root, 'GT-RAIN_test'))
        else:
            raise ValueError
        
        dataset_size = getattr(self.opt, split + '_dataset_size')
                
        names = []
        scenes = []
        rainy_dirs = []
        clean_dirs = []
        scene_indices = []
        wgt_or_not = []  # 表示一个图片是否有对应的gt
        last_index = 0
        if split == 'train':
            for root_path in root_paths:
                if 'WeatherStream' in root_path:
                    for scene in os.listdir(root_path):
                        scene_path = os.path.join(root_path, scene)
                        scene_length = 0
                        # if dataset_size == 'all': 
                        img_paths = os.listdir(scene_path)
                        img_paths = sorted(img_paths)
                        for img_path in img_paths:
                            if 'degraded' in img_path:
                                scene_length += 1
                                names.append(img_path)
                                rainy_dirs.append(os.path.join(scene_path, img_path))
                                if 'GT-Rain_crop1' in root_path:
                                    wgt_or_not.append(False)
                                else:
                                    wgt_or_not.append(True)
                                clean_dirs.append(os.path.join(scene_path, 'gt.png'))
                        # else:
                        #     img_paths = os.listdir(scene_path)[:int(dataset_size)]
                        #     # img_paths.sort(key=lambda x:int(x[-7:-4]))
                        #     for img_path in img_paths:
                        #         if 'degraded' in img_path:
                        #             scene_length += 1
                        #             names.append(img_path)
                        #             rainy_dirs.append(os.path.join(scene_path, img_path))
                        #             if 'GT-Rain_crop1' in root_path:
                        #                 wgt_or_not.append(False)
                        #             else:
                        #                 wgt_or_not.append(True)
                        #             clean_dirs.append(os.path.join(scene_path, 'gt.png'))       
                        scene_indices.append(list(range(last_index, last_index + scene_length)))
                        scenes.append(scene)
                        last_index += scene_length
                else:
                    for scene in os.listdir(root_path):
                        if scene == 'Gurutto_1-2':
                            continue 
                        scene_path = os.path.join(root_path, scene)
                        scene_length = 0
                        if dataset_size == 'all': 
                            img_paths = os.listdir(scene_path)
                            img_paths.sort(key=lambda x:int(x[-7:-4]))
                            for img_path in img_paths:
                                if img_path[-9] == 'R':
                                    scene_length += 1
                                    names.append(img_path)
                                    rainy_dirs.append(os.path.join(scene_path, img_path))
                                    if 'GT-Rain_crop1' in root_path:
                                        wgt_or_not.append(False)
                                    else:
                                        wgt_or_not.append(True)
                                    if scene == 'Gurutto_1-2':
                                        clean_dirs.append(os.path.join(scene_path, img_path[:-9] + 'C' + img_path[-8:]))
                                    else:
                                        clean_dirs.append(os.path.join(scene_path, img_path[:-9] + 'C-000.png'))
                        # else:
                        #     img_paths = os.listdir(scene_path)[:int(dataset_size)]
                        #     img_paths.sort(key=lambda x:int(x[-7:-4]))
                        #     for img_path in img_paths:
                        #         if img_path[-9] == 'R':
                        #             scene_length += 1
                        #             names.append(img_path)
                        #             rainy_dirs.append(os.path.join(scene_path, img_path))
                        #             if 'GT-Rain_crop1' in root_path:
                        #                 wgt_or_not.append(False)
                        #             else:
                        #                 wgt_or_not.append(True)
                        #             if scene == 'Gurutto_1-2':
                        #                 clean_dirs.append(os.path.join(scene_path, img_path[:-9] + 'C' + img_path[-8:]))
                        #             else:
                        #                 clean_dirs.append(os.path.join(scene_path, img_path[:-9] + 'C-000.png')) 
                        # import sys
                        # print(last_index, last_index + scene_length) 
                        # sys.stdout.flush()         
                        scene_indices.append(list(range(last_index, last_index + scene_length)))
                        scenes.append(scene)
                        last_index += scene_length
        else:
            for root_path in root_paths:
                if 'WeatherStream_test' in root_path:
                    for scene in os.listdir(root_path):
                        scene_path = os.path.join(root_path, scene)
                        if 'snow' in scene:
                            prefix = 'snow'
                        elif 'rain' in scene:
                            prefix = 'rain'
                        elif 'fog' in scene:
                            prefix = 'fog'
                            
                        scene_length = 0
                        if dataset_size == 'all': 
                            img_paths = os.listdir(scene_path)
                            img_paths = sorted(img_paths)
                            for img_path in img_paths:
                                if 'degraded' in img_path:
                                    scene_length += 1
                                    names.append(img_path)
                                    rainy_dirs.append(os.path.join(scene_path, img_path))
                                    clean_dirs.append(os.path.join(scene_path, prefix + '_gt.png'))
                                    if 'GT-Rain_crop1' in root_path:
                                        wgt_or_not.append(False)
                                    else:
                                        wgt_or_not.append(True)
                        else:
                            img_paths = os.listdir(scene_path)[:int(dataset_size)]
                            for img_path in img_paths:
                                if 'degraded' in img_path:
                                    scene_length += 1
                                    names.append(img_path)
                                    rainy_dirs.append(os.path.join(scene_path, img_path))
                                    clean_dirs.append(os.path.join(scene_path, prefix + '_gt.png'))
                                    if 'GT-Rain_crop1' in root_path:
                                        wgt_or_not.append(False)
                                    else:
                                        wgt_or_not.append(True) 
                        scene_indices.append(list(range(last_index, last_index + scene_length)))
                        scenes.append(scene)
                        last_index += scene_length
                else:
                    for scene in os.listdir(root_path):  
                        scene_path = os.path.join(root_path, scene)
                        scene_length = 0
                        # if dataset_size == 'all': 
                        img_paths = os.listdir(scene_path)
                        for img_path in img_paths:
                            if img_path[-9] == 'R':
                                scene_length += 1
                                names.append(img_path)
                                rainy_dirs.append(os.path.join(scene_path, img_path))
                                if 'GT-Rain_crop1' in root_path:
                                    wgt_or_not.append(False)
                                else:
                                    wgt_or_not.append(True)
                                clean_dirs.append(os.path.join(scene_path, img_path[:-9] + 'C-000.png'))
                        # else:
                        #     img_paths = os.listdir(scene_path)[:int(dataset_size)]
                        #     for img_path in img_paths:
                        #         if img_path[-9] == 'R':
                        #             scene_length += 1
                        #             names.append(img_path)
                        #             rainy_dirs.append(os.path.join(scene_path, img_path))
                        #             if 'GT-Rain_crop1' in root_path:
                        #                 wgt_or_not.append(False)
                        #             else:
                        #                 wgt_or_not.append(True)
                        #             clean_dirs.append(os.path.join(scene_path, img_path[:-9] + 'C-000.png'))  
                                            
                        scene_indices.append(list(range(last_index, last_index + scene_length)))
                        scenes.append(scene)
                        last_index += scene_length

        return names, scenes, rainy_dirs, clean_dirs, scene_indices, wgt_or_not

    def get_scene_indices(self):
        return self.scenes, self.scene_indices

    def _crop_patch(self, rainy_img, clean_img, ph, pw):
        
        rainy_img = rainy_img[..., ph:ph+self.patch_size, pw:pw+self.patch_size]
        clean_img = clean_img[..., ph:ph+self.patch_size, pw:pw+self.patch_size]
        return rainy_img, clean_img
                
    def _crop_center(self, rainy_img, clean_img, p=256):
        hh, ww = rainy_img.shape[-2:]
        begin_h, begin_w = hh // 2 - self.patch_size // 2, ww // 2 - self.patch_size // 2
        rainy_img = rainy_img[..., begin_h:begin_h+self.patch_size, begin_w:begin_w+self.patch_size]
        clean_img = clean_img[..., begin_h:begin_h+self.patch_size, begin_w:begin_w+self.patch_size]
        return rainy_img, clean_img

    def _flip(self, rainy_img, clean_img, aug):
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
