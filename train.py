#-*- encoding: UTF-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

import cv2
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import math
import sys
import random
import torch.multiprocessing as mp
import torch.nn as nn
import lpips
import os
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from util.util import calc_lpips 
from collections import OrderedDict
import shutil


cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

def setup_seed(seed=0):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
	setup_seed(seed=0)
	
	opt = TrainOptions().parse()
	
	source_path = os.path.abspath(r'../derain')
	target_path = os.path.abspath(r'../checkpoints/' + opt.name + '/code/')

	if not os.path.exists(target_path):
		os.makedirs(target_path)

	if os.path.exists(source_path):
		# 如果目标路径存在原文件夹的话就先删除
		shutil.rmtree(target_path)

	shutil.copytree(source_path, target_path)
	print('copy code finished!')

	loss_fn_alex_1 = lpips.LPIPS(net='alex', version='0.1')
	model = create_model(opt)
	model.setup(opt)

	dataset_train = create_dataset(opt.dataset_name, 'train', opt)
	dataset_size_train = len(dataset_train)
	print('The number of training images = %d' % dataset_size_train)
	dataset_val = create_dataset(opt.dataset_name, 'val', opt)
	dataset_size_val = len(dataset_val)
	print('The number of val images = %d' % dataset_size_val)

	visualizer = Visualizer(opt)
	total_iters = ((model.start_epoch * (dataset_size_train // opt.batch_size)) \
					// opt.print_freq) * opt.print_freq

	psnr = [0.0] * dataset_size_val
	ssim = [0.0] * dataset_size_val
	lpipses = [0.0] * dataset_size_val
	
	if model.start_epoch == 0 and opt.lr_policy == 'warmup':
		model.update_before_iter()
		# scheduler.step()
	for epoch in range(model.start_epoch + 1, opt.niter + opt.niter_decay + 1):
		# training
		epoch_start_time = time.time()
		epoch_iter = 0
		model.train()

		iter_data_time = iter_start_time = time.time()
		for i, data in enumerate(dataset_train):
			if total_iters % opt.print_freq == 0:
				t_data = time.time() - iter_data_time
			total_iters += 1 #opt.batch_size
			epoch_iter += 1; 
			model.set_input(data, epoch); 
			model.optimize_parameters(epoch); 

			if total_iters % opt.print_freq == 0:
				# print(total_iters, opt.print_freq)
				losses = model.get_current_losses()
				t_comp = (time.time() - iter_start_time)
				visualizer.print_current_losses(
					epoch, epoch_iter, losses, t_comp, t_data, total_iters)
				iter_start_time = time.time()
			iter_data_time = time.time()
			sys.stdout.flush()
		
		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d'
				  % (epoch, total_iters))
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %.3f sec'
			  % (epoch, opt.niter + opt.niter_decay,
				 time.time() - epoch_start_time))
		model.update_learning_rate()

		# val
		if opt.calc_metrics:
			model.eval()
			val_iter_time = time.time()
			tqdm_val = tqdm(dataset_val, ncols=85)
			
			time_val = 0
			for i, data in enumerate(tqdm_val):
				torch.cuda.empty_cache()
				model.set_input(data, -1)
				time_val_start = time.time()
				# with torch.no_grad():
				model.test()
				time_val += time.time() - time_val_start
				res = model.get_current_visuals()
				lpipses[i] = calc_lpips(res['clean_img'], res['derained_img'], loss_fn_alex_1, 'cuda:' + str(opt.gpu_ids[0]))
				derained = np.array(res['derained_img'][0].cpu()).transpose((1, 2, 0)) / 255.
				clean = np.array(res['clean_img'][0].cpu()).transpose((1, 2, 0)) / 255.
				psnr[i] = calc_psnr(clean, derained, data_range=1.)
				ssim[i] = calc_ssim(clean, derained, multichannel=True)

				if opt.save_imgs:
					save_dir_rgb = os.path.join('../checkpoints', opt.name, 'train_epoch_' + str(epoch), 'rgb_out', data['file_name'][0][:-17])
					os.makedirs(save_dir_rgb, exist_ok=True)
					out_img = np.array(res['derained_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
					cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'd.png'), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
					# rainy_img = np.array(res['rainy_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
					# print('存储图片时的大小', rainy_img.shape)
					# cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'r.png'), cv2.cvtColor(rainy_img, cv2.COLOR_RGB2BGR))
					clean_img = np.array(res['clean_img'][0].cpu()).astype(np.uint8).transpose((1, 2, 0))
					cv2.imwrite(os.path.join(save_dir_rgb, data['file_name'][0][:-4] + 'c.png'), cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))

			visualizer.print_psnr(epoch, opt.niter + opt.niter_decay, time_val, np.mean(psnr))
			visualizer.print_ssim(epoch, opt.niter + opt.niter_decay, time_val, np.mean(ssim))
			visualizer.print_lpips(epoch, opt.niter + opt.niter_decay, time_val, np.mean(lpipses))


		torch.cuda.empty_cache()
		sys.stdout.flush()


	