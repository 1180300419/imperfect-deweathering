import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
import struct
import cv2


def save_bin(filepath, arr):
    '''
        save 2-d numpy array to '.bin' files with uint16

    @param filepath:
        expected file path to store data

    @param arr:
        2-d numpy array

    @return:
        None

    '''

    arr = np.round(arr).astype('uint16')
    arr = np.clip(arr, 0, 1023)
    height, width = arr.shape

    with open(filepath, 'wb') as fp:
        fp.write(struct.pack('<HH', width, height))
        arr.tofile(fp)

# def copy_clean_img(out_bayer, clean_img):
#     out_bayer[0::4, 3::4] = clean_img[0::4, 3::4]
#     out_bayer[3::4, 0::4] = clean_img[3::4, 0::4]
#     return out_bayer

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
    split = opt.split
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, split, opt)
        print(len(dataset))
        datasets[dataset_name] = tqdm(dataset)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        # log_dir = '%s/%s/logs/log_epoch_%d.txt' % (
        #         opt.checkpoints_dir, opt.name, load_iter)
        # os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
        # f = open(log_dir, 'a')

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            psnr = [[0.0] * dataset_size_test,  [0.0] * dataset_size_test]
            time_val = 0
            print( dataset_size_test)
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                if 'real' in split:
                    i = -2 
                model.set_input(data, i)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()

                if opt.calc_metrics:
                    psnr[0][i] = calc_psnr(res['gt_raw_img'], res['data_out_raw'])
                    psnr[1][i] = calc_psnr(res['gt_rgb_img'], res['data_out_rgb'])
                
                if opt.save_imgs:
                    file_name_prefix = data['fname'][0]
                    folder_dir_raw = './ckpt/%s/%s/output_raw' % (opt.name, split)  
                    os.makedirs(folder_dir_raw, exist_ok=True)
                    save_dir_raw = '%s/%s.bin' % (folder_dir_raw, file_name_prefix[:-4])
                    # dataset_test.imio.write(np.array(res['data_out'][0].cpu()).astype(np.float16), save_dir)
                    data_out = np.array(res['data_out_raw'][0][0].cpu()).astype(np.float32)
                    data_out = (data_out ) * (1023-64) + 64
                    # print(res['noise_img'].shape)
                    # clean_data =  np.array(res['clean_img'][0][0].cpu()).astype(np.float32)
                    # clean_data = clean_data * (1023 - 64) + 64
                    # data_out = copy_clean_img(data_out, clean_data)
                    save_bin(save_dir_raw, data_out)
                    if 'real' not in split:
                        folder_dir_rgb = './ckpt/%s/%s/output_rgb' % (opt.name, split)
                        save_dir_rgb = '%s/%s.png' % (folder_dir_rgb, file_name_prefix[:-4])
                        os.makedirs(folder_dir_rgb, exist_ok=True)
                        # print(res['data_out_rgb'].shape)        torch.Size([1, 1200, 1800, 3])
                        # dataset_test.imio.write(np.array(res['data_out_rgb'][0].cpu()).astype(np.uint8), save_dir_rgb)
                        rgb = res['data_out_rgb'].permute(0, 2, 3, 1)
                        cv2.imwrite(save_dir_rgb, cv2.cvtColor(np.array(rgb[0].cpu()).astype(np.uint8), cv2.COLOR_RGB2BGR))


            avg_psnr_raw = '%.2f'%np.mean(psnr[0])
            avg_psnr_rgb = '%.2f'%np.mean(psnr[1])

            # f.write('dataset: %s, PSNR: %s, Time: %.3f sec.\n'
            #         % (dataset_name, avg_psnr, time_val))
            print('Time: %.3f s AVG Time: %.3f ms PSNR_RAW: %s PSNR_RGB: %s \n' % (time_val, time_val/dataset_size_test*1000, avg_psnr_raw, avg_psnr_rgb))
        #     f.flush()
        #     f.write('\n')
        # f.close()
    for dataset in datasets:
        datasets[dataset].close()


