'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-09-16 12:39:08
LastEditors: Liu Xiaohui
LastEditTime: 2022-10-16 20:08:53
'''
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
import cv2


if __name__ == '__main__':
    opt = TestOptions().parse()
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
        # print('before ')
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        # print('before reloop')
        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            # time_val = 0
            # print(dataset_size_test)
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                # time_val += time.time() - time_val_start
                res = model.get_current_visuals()

                if opt.save_imgs:
                    save_dir_rgb = os.path.join('../checkpoints', opt.name, 'test_epoch_' + str(opt.load_iter), 'inter_rgb_out')
                    os.makedirs(save_dir_rgb, exist_ok=True)
                    out_img = np.array(res['derained_img'][0].cpu()).round().astype(np.uint8).transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + '_gt.png'), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    for dataset in datasets:
        datasets[dataset].close()

