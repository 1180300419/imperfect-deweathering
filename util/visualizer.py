'''
Descripttion: 
version: 
Author: Liu Xiaohui
Date: 2022-09-27 12:50:49
LastEditors: Liu Xiaohui
LastEditTime: 2022-10-16 17:19:25
'''
import numpy as np
from os.path import join
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
from functools import partial
from functools import wraps
import time

def write_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(30):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                print('%s OSError' % str(args))
                time.sleep(1)
        return ret
    return wrapper

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        if opt.isTrain:
            self.name = opt.name
            self.save_dir = join(opt.checkpoints_dir, opt.name, 'log')
            self.writer = SummaryWriter(logdir=join(self.save_dir))
        else:
            self.name = '%s_%s_%d' % (
                opt.name, opt.dataset_name, opt.load_iter)
            self.save_dir = join(opt.checkpoints_dir, opt.name)
            if opt.save_imgs:
                self.writer = SummaryWriter(logdir=join(
                    self.save_dir, 'ckpts', self.name))

    @write_until_success
    def display_current_results(self, phase, visuals, iters):
        for k, v in visuals.items():
            v = v.cpu()
            self.writer.add_image('%s/%s'%(phase, k), v[0]/255., iters)
        self.writer.flush()

    @write_until_success
    def print_current_losses(self, epoch, iters, losses,
                             t_comp, t_data, total_iters):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' \
                  % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.4e ' % (k, v)
            self.writer.add_scalar('loss/%s'%k, v, total_iters)
        print(message)
    
    @write_until_success
    def print_psnr(self, epoch, total_epoch, time_val, mean_psnr):
        self.writer.add_scalar('val/psnr', mean_psnr, epoch)
        print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t PSNR: %f'
                % (epoch, total_epoch, time_val, mean_psnr))

    @write_until_success
    def print_ssim(self, epoch, total_epoch, time_val, mean_ssim):
        self.writer.add_scalar('val/ssim', mean_ssim, epoch)
        print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t SSIM: %f'
                % (epoch, total_epoch, time_val, mean_ssim))
    
    @write_until_success
    def print_lpips(self, epoch, total_epoch, time_val, mean_lpips):
        self.writer.add_scalar('val/lpips', mean_lpips, epoch)
        print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t LPIPS: %f'
                % (epoch, total_epoch, time_val, mean_lpips))

    @write_until_success
    def print_kld(self, epoch, total_epoch, time_val, mean_kld):
        self.writer.add_scalar('val/kld', mean_kld, epoch)
        print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t KLD: %f'
                % (epoch, total_epoch, time_val, mean_kld))

    @write_until_success
    def print_m4(self, epoch, total_epoch, time_val, mean_M4):
        self.writer.add_scalar('val/M4', mean_M4, epoch)
        print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t M4: %f'
                % (epoch, total_epoch, time_val, mean_M4))
