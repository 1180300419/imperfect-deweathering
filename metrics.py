import numpy as np
import cv2
import torch
import math
from tqdm import tqdm
import lpips
import glob
from skimage.metrics import structural_similarity as ssim
import argparse
import sys
import os


def calc_psnr_np(sr, hr, range=255.):
	diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
	mse = np.power(diff, 2).mean()
	return -10 * math.log10(mse)

def lpips_norm(img):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (255. / 2.) - 1
	return torch.Tensor(img).to(device)

def calc_lpips(out, target, loss_fn_alex):
	lpips_out = lpips_norm(out)
	lpips_target = lpips_norm(target)
	LPIPS = loss_fn_alex(lpips_out, lpips_target)
	return LPIPS.detach().cpu().item()

def calc_metrics(out, target, loss_fn_alex):
	psnr = calc_psnr_np(out, target)
	SSIM = ssim(out, target, win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
	LPIPS = calc_lpips(out, target, loss_fn_alex)
	return np.array([psnr, SSIM, LPIPS], dtype=float)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Metrics for argparse')
	parser.add_argument('--name', type=str, required=True,
			            help='Name of the folder to save models and logs.')	
	parser.add_argument('--dataroot', type=str, default='/Data/dataset/GOPRO_Large/')
	parser.add_argument('--device', default="0")
	args = parser.parse_args()

	device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
	loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(device)

	root = sys.path[0]
	files = [
		root + '/ckpt/' + args.name,
	]
	
	for file in files:
		print('Start to measure images in %s...' % (file))
		metrics = np.zeros([112*2, 3])
		log_dir = '%s/log_metrics.txt' % (file)
		f = open(log_dir, 'a')
		i = 0

		for image_file in tqdm(os.listdir(args.dataroot + 'val/HR/')):
			gt = cv2.imread(args.dataroot + 'val/HR/' + image_file )[..., ::-1]
			# outputt = cv2.imread(file + '/outputt/' + image_file)[..., ::-1]
			# outputf = cv2.imread(file + '/outputf/' + image_file)[..., ::-1]
			outputs = cv2.imread(file + '/outputs/' + image_file)[..., ::-1]
			outputss = cv2.imread(file + '/outputss/' + image_file)[..., ::-1]
			output = 0.7* outputss + 0.3* outputs
			# output = np.mean(np.concatenate((outputf.reshape(np.newaxis,...), outputs.reshape(np.newaxis,...)), axis=0),axis=0)
			# output = cv2.imread(file + '/output/' + image_file)[..., ::-1]
			metrics[i, 0:3] = calc_metrics(output, gt, loss_fn_alex_v1)
			i = i + 1

		mean_metrics = np.mean(metrics, axis=0)
		
		print('\n        File        :\t %s \n' % (file))
		print('   Original    GT   :\t PSNR = %.3f, SSIM = %.4f, LPIPS = %.3f \n' 
		        % (mean_metrics[0], mean_metrics[1], mean_metrics[2]))

		f.write('\n        File        :\t %s \n' % (file))
		f.write('   Original    GT   :\t PSNR = %.3f, SSIM = %.4f, LPIPS = %.3f \n' 
		        % (mean_metrics[0], mean_metrics[1], mean_metrics[2]))

		f.flush()
		f.close()
	