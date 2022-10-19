# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from torch.nn import L1Loss
from torchvision import models
import lpips
from piq import MultiScaleSSIMLoss
from skimage import exposure
from IQA_pytorch import SSIM
import util.util as util
import math


def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(
			-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) \
		for x in range(window_size)])
	return gauss / gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(
		channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
	mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(
		img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(
		img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(
		img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
			   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)

	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)

	return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIMLoss(nn.Module):
	def __init__(self, window_size=11, size_average=True):
		super(SSIMLoss, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and \
				self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)

			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)

			self.window = window
			self.channel = channel

		return _ssim(img1, img2, window, self.window_size,
					 channel, self.size_average)

class VGG19(torch.nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		features = models.vgg19(pretrained=True).features
		self.relu1_1 = torch.nn.Sequential()
		self.relu1_2 = torch.nn.Sequential()

		self.relu2_1 = torch.nn.Sequential()
		self.relu2_2 = torch.nn.Sequential()

		self.relu3_1 = torch.nn.Sequential()
		self.relu3_2 = torch.nn.Sequential()
		self.relu3_3 = torch.nn.Sequential()
		self.relu3_4 = torch.nn.Sequential()

		self.relu4_1 = torch.nn.Sequential()
		self.relu4_2 = torch.nn.Sequential()
		self.relu4_3 = torch.nn.Sequential()
		self.relu4_4 = torch.nn.Sequential()

		self.relu5_1 = torch.nn.Sequential()
		self.relu5_2 = torch.nn.Sequential()
		self.relu5_3 = torch.nn.Sequential()
		self.relu5_4 = torch.nn.Sequential()

		for x in range(2):
			self.relu1_1.add_module(str(x), features[x])

		for x in range(2, 4):
			self.relu1_2.add_module(str(x), features[x])

		for x in range(4, 7):
			self.relu2_1.add_module(str(x), features[x])

		for x in range(7, 9):
			self.relu2_2.add_module(str(x), features[x])

		for x in range(9, 12):
			self.relu3_1.add_module(str(x), features[x])

		for x in range(12, 14):
			self.relu3_2.add_module(str(x), features[x])

		for x in range(14, 16):
			self.relu3_3.add_module(str(x), features[x])

		for x in range(16, 18):
			self.relu3_4.add_module(str(x), features[x])

		for x in range(18, 21):
			self.relu4_1.add_module(str(x), features[x])

		for x in range(21, 23):
			self.relu4_2.add_module(str(x), features[x])

		for x in range(23, 25):
			self.relu4_3.add_module(str(x), features[x])

		for x in range(25, 27):
			self.relu4_4.add_module(str(x), features[x])

		for x in range(27, 30):
			self.relu5_1.add_module(str(x), features[x])

		for x in range(30, 32):
			self.relu5_2.add_module(str(x), features[x])

		for x in range(32, 34):
			self.relu5_3.add_module(str(x), features[x])

		for x in range(34, 36):
			self.relu5_4.add_module(str(x), features[x])

		# don't need the gradients, just want the features
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		relu1_1 = self.relu1_1(x)
		relu1_2 = self.relu1_2(relu1_1)

		relu2_1 = self.relu2_1(relu1_2)
		relu2_2 = self.relu2_2(relu2_1)

		relu3_1 = self.relu3_1(relu2_2)
		relu3_2 = self.relu3_2(relu3_1)
		relu3_3 = self.relu3_3(relu3_2)
		relu3_4 = self.relu3_4(relu3_3)

		relu4_1 = self.relu4_1(relu3_4)
		relu4_2 = self.relu4_2(relu4_1)
		relu4_3 = self.relu4_3(relu4_2)
		relu4_4 = self.relu4_4(relu4_3)

		relu5_1 = self.relu5_1(relu4_4)
		relu5_2 = self.relu5_2(relu5_1)
		relu5_3 = self.relu5_3(relu5_2)
		relu5_4 = self.relu5_4(relu5_3)

		out = {
			'relu1_1': relu1_1,
			'relu1_2': relu1_2,

			'relu2_1': relu2_1,
			'relu2_2': relu2_2,

			'relu3_1': relu3_1,
			'relu3_2': relu3_2,
			'relu3_3': relu3_3,
			'relu3_4': relu3_4,

			'relu4_1': relu4_1,
			'relu4_2': relu4_2,
			'relu4_3': relu4_3,
			'relu4_4': relu4_4,

			'relu5_1': relu5_1,
			'relu5_2': relu5_2,
			'relu5_3': relu5_3,
			'relu5_4': relu5_4,
		}
		return out

class VGGLoss(nn.Module):
	def __init__(self):
		super(VGGLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = torch.nn.L1Loss()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		content_loss = 0.0
		# # content_loss += self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) * 0.1
		# # content_loss += self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) * 0.2
		content_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2']) * 1
		content_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2']) * 1
		content_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2']) * 2

		return content_loss / 4.

class TextureLoss(nn.Module):
	def __init__(self):
		super(TextureLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = torch.nn.L1Loss()

	def compute_gram(self, x):
		b, ch, h, w = x.size()
		f = x.view(b, ch, w * h)
		f_T = f.transpose(1, 2)
		G = f.bmm(f_T) / (h * w)
		return G

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		style_loss = 0.0
		style_loss += self.criterion(self.compute_gram(x_vgg['relu1_2']), self.compute_gram(y_vgg['relu1_2'])) * 0.2
		style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2'])) * 1
		style_loss += self.criterion(self.compute_gram(x_vgg['relu3_2']), self.compute_gram(y_vgg['relu3_2'])) * 1
		style_loss += self.criterion(self.compute_gram(x_vgg['relu4_2']), self.compute_gram(y_vgg['relu4_2'])) * 2
		style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2'])) * 5

		return style_loss * 0.007

class SWDLoss(nn.Module):
	def __init__(self):
		super(SWDLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = SWD()
		# self.SWD = SWDLocal()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		N, C, H, W = x.shape  # 192*192
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		swd_loss = 0.0
		swd_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'], k=H//4//p) * 1  # H//4=48
		swd_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'], k=H//8//p) * 1  # H//4=24
		swd_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'], k=H//16//p) * 2  # H//4=12

		return swd_loss * 8 / 100.0

class VGGStyleDiscriminator160(nn.Module):

	def __init__(self, num_in_ch=4, num_feat=64):
		super(VGGStyleDiscriminator160, self).__init__()
		# 12 * 4 * 128 * 128
		self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True) # 12 * 64 * 128 * 128
		self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False) # 12 * 64 * 64 * 64
		self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

		self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False) # 12 * 128 * 64 * 64
		self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
		self.conv1_1 = nn.Conv2d(
			num_feat * 2, num_feat * 2, 4, 2, 1, bias=False) # 12 * 128 * 32 * 32
		self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

		self.conv2_0 = nn.Conv2d(
			num_feat * 2, num_feat * 4, 3, 1, 1, bias=False) # 12 * 256 * 32 * 32
		self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
		self.conv2_1 = nn.Conv2d(
			num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)# 12 * 256 * 16 * 16
		self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

		self.conv3_0 = nn.Conv2d(
			num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)  # 12 * 512 * 16 * 16
		self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
		self.conv3_1 = nn.Conv2d(
			num_feat * 8, num_feat * 8, 4, 2, 1, bias=False) # 12 * 512 * 8 * 8
		self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

		self.conv4_0 = nn.Conv2d(
			num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)  # 12 * 512 * 8 * 8
		self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
		self.conv4_1 = nn.Conv2d(
			num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)  # 12 * 512 * 4 * 4
		self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
		# 6 * num_feat * 8 * 4 * 4
		self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
		self.linear2 = nn.Linear(100, 1)

		# activation function
		self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		# assert x.size(2) == 160 and x.size(3) == 160, (
		#     f'Input spatial size must be 160x160, '
		#     f'but received {x.size()}.')
		# print(x.shape)
		feat = self.lrelu(self.conv0_0(x))
		# print(feat.shape)
		# exit(0)
		feat = self.lrelu(self.bn0_1(
			self.conv0_1(feat)))  # output spatial size: (80, 80)

		feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
		feat = self.lrelu(self.bn1_1(
			self.conv1_1(feat)))  # output spatial size: (40, 40)

		feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
		feat = self.lrelu(self.bn2_1(
			self.conv2_1(feat)))  # output spatial size: (20, 20)

		feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
		feat = self.lrelu(self.bn3_1(
			self.conv3_1(feat)))  # output spatial size: (10, 10)

		feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
		feat = self.lrelu(self.bn4_1(
			self.conv4_1(feat)))  # output spatial size: (5, 5)
		
		feat = feat.view(feat.size(0), -1)
		# print(feat.shape)
		# exit(0)
		feat = self.lrelu(self.linear1(feat))
		out = self.linear2(feat)
		return out

class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
		
class AdversarialLoss(nn.Module):
	def __init__(self, gpu_ids=[], dist=False, gan_mode='RGAN', gan_k=3,
					lr_dis=1e-4, train_crop_size=40):

		super(AdversarialLoss, self).__init__()
		self.gan_mode = gan_mode
		self.gan_k = gan_k
		# self.device = torch.device('cpu' if use_cpu else 'cuda' +)
		self.discriminator = VGGStyleDiscriminator160(num_in_ch=4, num_feat=64).to(gpu_ids[0])
			
		self.optimizer = torch.optim.Adam(
				self.discriminator.parameters(),
				betas=(0, 0.9), eps=3e-5, lr=lr_dis
			)
		if self.gan_mode == 'RGAN':
			self.criterion_adv = GANLoss(gan_mode='vanilla').to(gpu_ids[0])
		else:
			self.criterion_adv = GANLoss(gan_mode=self.gan_mode).to(gpu_ids[0])
		if len(gpu_ids) > 1:
			self.discriminator = torch.nn.DataParallel(self.discriminator, gpu_ids)
			# self.criterion_adv = torch.nn.DataParallel(self.criterion_adv, gpu_ids)
	def set_requires_grad(self, nets, requires_grad=False):
		"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
		Parameters:
			nets (network list)   -- a list of networks
			requires_grad (bool)  -- whether the networks require gradients or not
		"""
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad
	
	def pack_bayer_img(self, bayer):
		n, c, h, w = bayer.shape
		out = bayer.new_zeros(n, 4, h // 2, w // 2)
		out[..., 0, :, :] = bayer[..., 0, 0::2, 0::2]
		out[..., 1, :, :] = bayer[..., 0, 0::2, 1::2]
		out[..., 2, :, :] = bayer[..., 0, 1::2, 0::2]
		out[..., 3, :, :] = bayer[..., 0, 1::2, 1::2]
		return out

	def forward(self, fake, real):
		fake = self.pack_bayer_img(fake)
		real = self.pack_bayer_img(real)
		# D Loss
		for _ in range(self.gan_k):
			self.set_requires_grad(self.discriminator, True)
			self.optimizer.zero_grad()
			# real
			d_fake = self.discriminator(fake).detach()
			d_real = self.discriminator(real)
			if self.gan_mode == 'RGAN':
				d_real_loss = self.criterion_adv(d_real - torch.mean(d_fake), True,
													is_disc=True) * 0.5
			elif self.gan_mode == 'lsgan':
				d_real_loss = self.criterion_adv(d_real, True, is_disc=True) * 0.5

			d_real_loss.backward()
			# fake
			d_fake = self.discriminator(fake.detach())
			if self.gan_mode == 'RGAN':
				d_fake_loss = self.criterion_adv(d_fake - torch.mean(d_real.detach()), False,
													is_disc=True) * 0.5
			elif self.gan_mode == 'lsgan':
				d_fake_loss = self.criterion_adv(d_fake, False, is_disc=True) * 0.5

			d_fake_loss.backward()
			loss_d = d_real_loss + d_fake_loss
			
			self.optimizer.step()

		# G Loss
		self.set_requires_grad(self.discriminator, False)
		d_real = self.discriminator(real).detach()
		d_fake = self.discriminator(fake)
		if self.gan_mode == 'RGAN':
			g_real_loss = self.criterion_adv(d_real - torch.mean(d_fake), False, is_disc=False) * 0.5
			g_fake_loss = self.criterion_adv(d_fake - torch.mean(d_real), True, is_disc=False) * 0.5
			loss_g = g_real_loss + g_fake_loss
		elif self.gan_mode == 'lsgan':
			loss_g = self.criterion_adv(d_fake, False, is_disc=False)
		# Generator loss
		return loss_g, loss_d

class SWD(nn.Module):
	def __init__(self):
		super(SWD, self).__init__()
		self.l1loss = torch.nn.L1Loss() 

	def forward(self, fake_samples, true_samples, k=0):
		N, C, H, W = true_samples.shape

		num_projections = C

		true_samples = true_samples.view(N, C, -1)
		fake_samples = fake_samples.view(N, C, -1)

		projections = torch.from_numpy(np.random.normal(size=(num_projections, C)).astype(np.float32))
		projections = torch.FloatTensor(projections).to(true_samples.device)
		projections = F.normalize(projections, p=2, dim=1)

		projected_true = projections @ true_samples
		projected_fake = projections @ fake_samples

		sorted_true, true_index = torch.sort(projected_true, dim=2)
		sorted_fake, fake_index = torch.sort(projected_fake, dim=2)
		return self.l1loss(sorted_true, sorted_fake).mean() 

class MultiLoss(nn.Module):
	def __init__(self):
		super(MultiLoss, self).__init__()
		self.l1 = L1Loss()
		self.swd = SWDLoss()
	
	def forward(self, sr, hr):
		loss_EDSR_L1 = 0
		loss_EDSR_SWD = 0
		for scale in [0.5, 1, 2, 4]:
			data_sr = nn.functional.interpolate(input=sr, scale_factor=scale/4, mode='bilinear', align_corners=True)
			data_hr = nn.functional.interpolate(input=hr, scale_factor=scale/4, mode='bilinear', align_corners=True)

			loss_EDSR_L1 = loss_EDSR_L1 + self.l1(data_sr, data_hr).mean() * scale
			loss_EDSR_SWD = loss_EDSR_SWD + self.swd(data_sr, data_hr).mean() * scale

		loss_EDSR_L1 = loss_EDSR_L1.mean() / 7.5
		loss_EDSR_SWD = loss_EDSR_SWD.mean() / 7.5
		return loss_EDSR_L1, loss_EDSR_SWD, loss_EDSR_L1 + loss_EDSR_SWD

class TVLoss(nn.Module):
	def __init__(self, TVLoss_weight=1):
		super(TVLoss,self).__init__()
		self.TVLoss_weight = TVLoss_weight

	def forward(self, x):
		batch_size = x.size()[0]
		h_x = x.size()[2]
		w_x = x.size()[3]
		count_h = self._tensor_size(x[:,:,1:,:])
		count_w = self._tensor_size(x[:,:,:,1:])
		h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
		w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
		return self.TVLoss_weight * 2 * (h_tv/count_h+w_tv/count_w) / batch_size

	def _tensor_size(self,t):
		return t.size()[1] * t.size()[2] * t.size()[3]

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class LPIPSLoss(nn.Module):
	def __init__(self, use_cpu=False, gpu_ids=[]):
		super(LPIPSLoss, self).__init__()
		self.device = torch.device('cpu' if use_cpu else 'cuda')
		# self.alex = lpips.LPIPS(net='alex', version='0.1').to(self.device)
		self.vgg = lpips.LPIPS(net='vgg', version='0.1').to(self.device)
		# self.set_requires_grad(self.alex)
		self.set_requires_grad(self.vgg)
		# self.alex.eval()
		self.vgg.eval()

	def set_requires_grad(self, nets, requires_grad=False):
			"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
			Parameters:
				nets (network list)   -- a list of networks
				requires_grad (bool)  -- whether the networks require gradients or not
			"""
			if not isinstance(nets, list):
				nets = [nets]
			for net in nets:
				if net is not None:
					for param in net.parameters():
						param.requires_grad = requires_grad
	
	def nor(self, img):
		return img / (255. / 2.) - 1
	
	def forward(self, sr, hr):
		# sr = self.nor(sr)
		# hr = self.nor(hr)

		# LPIPS = self.alex(sr, hr)
		LPIPS = self.vgg(sr, hr)
		loss = torch.mean(LPIPS)
		return LPIPS

def norm_min_max_distributuions(distributuions: torch.Tensor):
    max_ = max(torch.max(d.data) for d in distributuions)
    min_ = min(torch.min(d.data) for d in distributuions)

    norm_distributuions = (distributuions - min_) / (max_ - min_)
    return norm_distributuions

def triangular_histogram_with_linear_slope(inputs: torch.Tensor, t: torch.Tensor, delta: float):
    """
    Function that calculates a histogram from an article
    [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)
    Args:
        input (Tensor): tensor that contains the data
        t (Tensor): tensor that contains the nodes of the histogram
        delta (float): step in histogram
    """
    b,c,h,w = inputs.shape
    n = h*w
    inputs = inputs.reshape(b,c,n)
    t =  torch.unsqueeze(torch.unsqueeze(t,0),0)
    # print(inputs.unsqueeze(2).shape,t.unsqueeze(3).shape)
    # first condition of the second equation of the paper
    x = inputs.unsqueeze(2) - t.unsqueeze(3) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    a = torch.sum(x * m, dim=3) / (delta * n)
    # print(a.shape, torch.sum(a))
    # second condition of the second equation of the paper
    x = t.unsqueeze(2) - inputs.unsqueeze(3) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x < delta)] = 1
    b = torch.sum(x * m, dim=2) / (delta * n)
    # print(b.shape, torch.sum(b))
    return torch.add(a, b)

class KL_Loss(nn.Module):
    def __init__(self, left_edge=0,  right_edge=1023, n_bins=1024, sigma=1):
        super(KL_Loss, self).__init__()
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.n_bins = n_bins
        self.delta = (self.right_edge - self.left_edge) / (self.n_bins - 1)
        self.centers = torch.arange(self.left_edge, self.right_edge + self.delta, step=self.delta)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
    
    def forward(self, bayer_gt, bayer_out):
        # print('loss:',bayer_gt.shape) [b/#_of_gups, 1, h, w]
        score_channels = torch.zeros((2, 2), dtype=bayer_gt.dtype, device=bayer_gt.device)
        for i in range(2):
            for j in range(2):
                score_channels[i, j] = self.cal_kld_bayer(bayer_gt[...,i::2, j::2], bayer_out[...,i::2, j::2])

        return torch.mean(score_channels)
    
    def cal_kld_bayer(self, bayer_gt, bayer_out):
        h_gt = triangular_histogram_with_linear_slope(bayer_gt, self.centers, self.delta)
        h_out = triangular_histogram_with_linear_slope(bayer_out, self.centers, self.delta)
        bb, cc, ww, hh = bayer_gt.shape
        # print(bayer_gt.shape, bayer_out.shape) [b/#_of_gups, 1, h/2, w/2]
        min_val = torch.ones_like(h_out) / (ww*hh)
        h_gt = torch.where(h_gt != 0, h_gt, min_val)
        h_out = torch.where(h_out != 0, h_out, min_val)

        # KL_divergence: D_kl_fwd = sum{h(x) * log[h(x) / h_out(x)]}
        kl_fwd = torch.sum(h_gt  * (torch.log(h_gt) - torch.log(h_out)))
        kl_inv = torch.sum(h_out * (torch.log(h_out) - torch.log(h_gt)))
        # print((kl_fwd + kl_inv)/2)
        return (kl_fwd + kl_inv)/2

class ShiftMSSSIM(torch.nn.Module):
  """Shifted SSIM Loss """
  def __init__(self):
    super(ShiftMSSSIM, self).__init__()
    self.ssim = MultiScaleSSIMLoss(data_range=1.)

  def forward(self, est, gt):
    # shift images back into range (0, 1)
    est = est * 0.5 + 0.5
    gt = gt * 0.5 + 0.5
    return self.ssim(est, gt)

class MSSSIM(torch.nn.Module):
	"""Shifted SSIM Loss """
	def __init__(self):
		super(MSSSIM, self).__init__()
		self.ssim = MultiScaleSSIMLoss(data_range=1.)

	def forward(self, est, gt):
		return self.ssim(est, gt)

# Rain Robust Loss
# Code modified from: https://github.com/sthalles/SimCLR/blob/master/simclr.py

class RainRobustLoss(torch.nn.Module):
	"""Rain Robust Loss"""
	def __init__(self, batch_size, n_views, temperature=0.07):
		super(RainRobustLoss, self).__init__()
		self.batch_size = batch_size
		self.n_views = n_views
		self.temperature = temperature  # 0.25
		self.criterion = torch.nn.CrossEntropyLoss()

	def forward(self, features):
		logits, labels = self.info_nce_loss(features)
		return self.criterion(logits, labels)

	def info_nce_loss(self, features):
		labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
		labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
		labels = labels.to(features.device)

		features = F.normalize(features, dim=1) # 长度变成１

		similarity_matrix = torch.matmul(features, features.T)

		mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
		labels = labels[~mask].view(labels.shape[0], -1)
		# print('similarity_metrix: ', similarity_matrix.shape)  # 8*8
		# print('mask: ', mask.shape)  # 16*16
		similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
		positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

		# select only the negatives the negatives
		negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

		logits = torch.cat([positives, negatives], dim=1)  
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

		logits = logits / self.temperature
		return logits, labels


def rgb2gray(rgb):
  r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

class YL1Loss(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1loss = torch.nn.L1Loss()
  
  def forward(self, out, tar):
    y_out = rgb2gray(out)
    y_tar = rgb2gray(tar)
    return self.l1loss(y_out, y_tar)

# According to the Color Loss in https://ieeexplore.ieee.org/document/9565368
class ColorLoss(torch.nn.Module):
	def __init__(self):
		super(ColorLoss, self).__init__()

	def forward(self, in1, in2):
		assert in1.shape == in2.shape, "in1's shape is not equal to in2's shape"
		b, c, h, w = in1.shape

		in1 = in1.permute((0, 2, 3, 1))
		in2 = in2.permute((0, 2, 3, 1))

		in1 = F.normalize(in1, dim=3)
		in2 = F.normalize(in2, dim=3)

		out = in1 * in2
		out = out.sum(dim=3)
		out = out.resize(b, h * w)
		out = torch.mean(out, dim=1)
		return (1 - out).mean()

def hist_matching(src, ref):
	# determine if we are performing multichannel histogram matching
	# and then perform histogram matching itself
	matched = exposure.match_histograms(src, ref, multichannel=True)
	return matched


class L1_Charbonnier_loss(torch.nn.Module):
	"""L1 Charbonnierloss."""
	def __init__(self):
		super(L1_Charbonnier_loss, self).__init__()
		self.eps = 1e-6

	def forward(self, X, Y):
		diff = torch.add(X, -Y)
		loss = torch.sqrt(diff * diff + self.eps)
		return loss

def compute_meshgrid(shape):
	N, C, H, W = shape
	rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
	cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

	feature_grid = torch.meshgrid(rows, cols)
	feature_grid = torch.stack(feature_grid).unsqueeze(0)
	feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

	return feature_grid

# TODO: Considering avoiding OOM.
def compute_l2_distance(x, y):
	N, C, H, W = x.size()
	x_vec = x.view(N, C, -1)
	y_vec = y.view(N, C, -1)
	x_s = torch.sum(x_vec ** 2, dim=1)
	y_s = torch.sum(y_vec ** 2, dim=1)

	A = y_vec.transpose(1, 2) @ x_vec
	dist = y_s - 2 * A + x_s.transpose(0, 1)
	dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
	dist = dist.clamp(min=0.)

	return dist

def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde

def compute_cx(dist_tilde, band_width):
	w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
	cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
	return cx

def compute_relative_distance(dist_raw):
	dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
	dist_tilde = dist_raw / (dist_min + 1e-5)
	return dist_tilde

def compute_cosine_distance(x, y):
	# mean shifting by channel-wise mean of `y`.
	y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
	x_centered = x - y_mu
	y_centered = y - y_mu

	# L2 normalization
	x_normalized = F.normalize(x_centered, p=2, dim=1)
	y_normalized = F.normalize(y_centered, p=2, dim=1)

	# channel-wise vectorization
	N, C, *_ = x.size()
	x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
	y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

	# consine similarity
	cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
							y_normalized)  # (N, H*W, H*W)

	# convert to distance
	dist = 1 - cosine_sim

	return dist

# TODO: Considering avoiding OOM.
def compute_l1_distance(x: torch.Tensor, y: torch.Tensor):
	N, C, H, W = x.size()
	x_vec = x.view(N, C, -1)
	y_vec = y.view(N, C, -1)

	dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
	dist = dist.sum(dim=1).abs()
	dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
	dist = dist.clamp(min=0.)

	return dist


class Contextual_Bilateral_Loss(torch.nn.Module):
	"""
	Computes Contextual Bilateral (CoBi) Loss between x and y,
		proposed in https://arxiv.org/pdf/1905.05169.pdf.
	Parameters
	---
	x : torch.Tensor
		features of shape (N, C, H, W).
	y : torch.Tensor
		features of shape (N, C, H, W).
	band_width : float, optional
		a band-width parameter used to convert distance to similarity.
		in the paper, this is described as :math:`h`.
	loss_type : str, optional
		a loss type to measure the distance between features.
		Note: `l1` and `l2` frequently raises OOM.
	Returns
	---
	cx_loss : torch.Tensor
		contextual loss between x and y (Eq (1) in the paper).
	k_arg_max_NC : torch.Tensor
		indices to maximize similarity over channels.
	"""
	def __init__(self, weight_sp=0.1, band_width=1., loss_type='cosine'):
		super(Contextual_Bilateral_Loss, self).__init__()
		self.weight_sp=weight_sp
		self.band_width=band_width

		LOSS_TYPES = ['cosine', 'l1', 'l2']
		assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
		self.loss_type=loss_type

	def forward(self, x, y):

		assert x.size() == y.size(), 'input tensor must have the same size.'

		# spatial loss
		grid = compute_meshgrid(x.shape).to(x.device)
		dist_raw = compute_l2_distance(grid, grid)
		dist_tilde = compute_relative_distance(dist_raw)
		cx_sp = compute_cx(dist_tilde, self.band_width)

		# feature loss
		if self.loss_type == 'cosine':
			dist_raw = compute_cosine_distance(x, y)
		elif self.loss_type == 'l1':
			dist_raw = compute_l1_distance(x, y)
		elif self.loss_type == 'l2':
			dist_raw = compute_l2_distance(x, y)
		dist_tilde = compute_relative_distance(dist_raw)
		cx_feat = compute_cx(dist_tilde, self.band_width)

		# combined loss
		with torch.no_grad():
			cx_combine = (1. - self.weight_sp) * cx_feat + self.weight_sp * cx_sp

			k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

			cx = k_max_NC.mean(dim=1)
			cx_loss = torch.mean(-torch.log(cx + 1e-5))

		return cx_loss

#####
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        if self.reduction == 'mean':
            loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        else:
            loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class HuberLoss(nn.Module):
    """Huber Loss (L1)"""
    def __init__(self, delta=1e-2, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, x, y):
        abs_diff = torch.abs(x - y)
        q_term = torch.min(abs_diff, torch.full_like(abs_diff, self.delta))
        l_term = abs_diff - q_term
        if self.reduction == 'mean':
            loss = torch.mean(0.5 * q_term ** 2 + self.delta * l_term)
        else:
            loss = torch.sum(0.5 * q_term ** 2 + self.delta * l_term)
        return loss


class TVLoss(nn.Module):
    """Total Variation Loss"""
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, w=4, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)

    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class StyleLoss(nn.Module):
    """Style Loss"""
    def __init__(self):
        super(StyleLoss, self).__init__()

    @staticmethod
    def gram_matrix(self, x):
        B, C, H, W = x.size()
        features = x.view(B * C, H * W)
        G = torch.mm(features, features.t())
        return G.div(B * C * H * W)

    def forward(self, input, target):
        G_i = self.gram_matrix(input)
        G_t = self.gram_matrix(target).detach()
        loss = F.mse_loss(G_i, G_t)
        return loss


class GradientPenaltyLoss(nn.Module):
    """Gradient Penalty Loss"""
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss

class PyramidLoss(nn.Module):
    """Pyramid Loss"""
    def __init__(self, num_levels=3, pyr_mode='gau', loss_mode='l1', reduction='mean'):
        super(PyramidLoss, self).__init__()
        self.num_levels = num_levels
        self.pyr_mode = pyr_mode
        self.loss_mode = loss_mode
        assert self.pyr_mode == 'gau' or self.pyr_mode == 'lap'
        if self.loss_mode == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif self.loss_mode == 'l2':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.loss_mode == 'hb':
            self.loss = HuberLoss(reduction=reduction)
        elif self.loss_mode == 'cb':
            self.loss = CharbonnierLoss(reduction=reduction)
        else:
            raise ValueError()

    def forward(self, x, y):
        B, C, H, W = x.shape
        device = x.device
        gauss_kernel = util.gauss_kernel(size=5, device=device, channels=C)
        if self.pyr_mode == 'gau':
            pyr_x = util.gau_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
            pyr_y = util.gau_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        else:
            pyr_x = util.lap_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
            pyr_y = util.lap_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        loss = 0
        for i in range(self.num_levels):
            loss += self.loss(pyr_x[i], pyr_y[i])
        return loss

class LapPyrLoss(nn.Module):
    """Pyramid Loss"""
    def __init__(self, num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean'):
        super(LapPyrLoss, self).__init__()
        self.num_levels = num_levels
        self.lf_mode = lf_mode
        self.hf_mode = hf_mode
        if lf_mode == 'ssim':
            self.lf_loss = SSIM(channels=1)
        elif lf_mode == 'cb':
            self.lf_loss = CharbonnierLoss(reduction=reduction)
        else:
            raise ValueError()
        if hf_mode == 'ssim':
            self.hf_loss = SSIM(channels=1)
        elif hf_mode == 'cb':
            self.hf_loss = CharbonnierLoss(reduction=reduction)
        else:
            raise ValueError()

    def forward(self, x, y):
        B, C, H, W = x.shape
        device = x.device
        gauss_kernel = util.gauss_kernel(size=5, device=device, channels=C)
        pyr_x = util.laplacian_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
        pyr_y = util.laplacian_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        loss = self.lf_loss(pyr_x[-1], pyr_y[-1])
        for i in range(self.num_levels - 1):
            loss += self.hf_loss(pyr_x[i], pyr_y[i])
        return loss

class AlignedL1(nn.Module):
	""" Computes L2 error after performing spatial and color alignment of the input image to GT"""
	def __init__(self, alignment_net, sr_factor=1, boundary_ignore=None):
		super().__init__()
		self.sca = SpatialColorAlignment(alignment_net, sr_factor)
		self.boundary_ignore = boundary_ignore
		self.l1loss = torch.nn.L1Loss()

	def forward(self, pred, gt):
		pred_warped_m, valid = self.sca(pred, gt)

		# Ignore boundary pixels if specified
		if self.boundary_ignore is not None:
			pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
							self.boundary_ignore:-self.boundary_ignore]
			gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

			valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

		# Estimate MSE
		# mse = F.mse_loss(pred_warped_m, gt, reduction='none')

		# eps = 1e-12
		# elem_ratio = mse.numel() / valid.numel()
		# mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
		loss = self.l1loss(gt, pred_warped_m)
		return loss

def get_gaussian_kernel(sd, ksz=None):
    """ Returns a 2D Gaussian kernel with standard deviation sd """
    if ksz is None:
        ksz = int(4 * sd + 1)

    assert ksz % 2 == 1
    K = gauss_2d(ksz, sd, (0.0, 0.0), density=True)
    K = K / K.sum()
    return K.unsqueeze(0), 

def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    """ Returns a 2-D Gaussian """
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    if isinstance(sz, int):
        sz = (sz, sz)

    if isinstance(center, (list, tuple)):
        center = torch.tensor(center).view(1, 2)

    return gauss_1d(sz[0], sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1], sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)

def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    """ Returns a 1-D Gaussian """
    k = torch.arange(-(sz-1)/2, (sz+1)/2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0/(2*sigma**2) * (k - center.reshape(-1, 1))**2)
    if density:
        gauss /= math.sqrt(2*math.pi) * sigma
    return gauss

def warp(feat, flow, mode='bilinear', padding_mode='zeros'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow im1 --> im2
    input flow must be in format (x, y) at every pixel
    feat: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow (x, y)
    """
    B, C, H, W = feat.size()

    # mesh grid
    rowv, colv = torch.meshgrid([torch.arange(0.5, H + 0.5, device=feat.device),
                                 torch.arange(0.5, W + 0.5, device=feat.device)])
    grid = torch.stack((colv, rowv), dim=0).unsqueeze(0).float()
    grid = grid + flow

    # scale grid to [-1,1]
    grid_norm_c = 2.0 * grid[:, 0] / W - 1.0
    grid_norm_r = 2.0 * grid[:, 1] / H - 1.0

    grid_norm = torch.stack((grid_norm_c, grid_norm_r), dim=1)

    grid_norm = grid_norm.permute(0, 2, 3, 1)

    output = F.grid_sample(feat, grid_norm, mode=mode, padding_mode=padding_mode)

    return output

def apply_kernel(im, ksz, kernel):
    """ apply the provided kernel on input image """
    shape = im.shape
    im = im.view(-1, 1, *im.shape[-2:])

    pad = [ksz // 2, ksz // 2, ksz // 2, ksz // 2]
    im = F.pad(im, pad, mode='reflect')
    im_out = F.conv2d(im, kernel).view(shape)
    return im_out

def match_colors(im_ref, im_q, im_test, ksz, gauss_kernel):
    """ Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
        im_test
    """
    gauss_kernel = gauss_kernel.to(im_ref.device)
    bi = 5

    # Apply Gaussian smoothing
    im_ref_mean = apply_kernel(im_ref, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()
    im_q_mean = apply_kernel(im_q, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()

    im_ref_mean_re = im_ref_mean.view(*im_ref_mean.shape[:2], -1)
    im_q_mean_re = im_q_mean.view(*im_q_mean.shape[:2], -1)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        c = torch.lstsq(ir.t(), iq.t())
        c = c.solution[:3]
        c_mat_all.append(c)

    c_mat = torch.stack(c_mat_all, dim=0)
    im_q_mean_conv = torch.matmul(im_q_mean_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_q_mean_conv = im_q_mean_conv.view(im_q_mean.shape)

    err = ((im_q_mean_conv - im_ref_mean) * 255.0).norm(dim=1)

    thresh = 20

    # If error is larger than a threshold, ignore these pixels
    valid = err < thresh

    pad = (im_q.shape[-1] - valid.shape[-1]) // 2
    pad = [pad, pad, pad, pad]
    valid = F.pad(valid, pad)

    upsample_factor = im_test.shape[-1] / valid.shape[-1]
    valid = F.interpolate(valid.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear')
    valid = valid > 0.9

    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)

    return im_t_conv, valid

class SpatialColorAlignment(nn.Module):
    def __init__(self, alignment_net, sr_factor=4):
        super().__init__()
        self.sr_factor = sr_factor
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.alignment_net.to(device)
        self.gauss_kernel = self.gauss_kernel.to(device)

    def forward(self, pred, gt):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        # sr_factor = self.sr_factor
        # ds_factor = 1.0 / float(2.0 * sr_factor)
        # flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear') * ds_factor

        # burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        # burst_0_warped = warp(burst_0, flow_ds)
        # frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear')

        # # Match the colorspace between the prediction and ground truth
        # pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz, self.gauss_kernel)

        return pred_warped
