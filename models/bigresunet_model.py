from email.policy import default
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch
import functools
from . import networks as N
from . import BaseModel as BaseModel
from . import losses as L


class BIGRESUNETModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		parser.add_argument('--data_section', type=str, default='-1-1')
		parser.add_argument('--ngf', type=int, default=64)
		parser.add_argument('--n_blocks', type=int, default=9)
		parser.add_argument('--norm_layer_type', type=str, default='batch')
		parser.add_argument('--upsample_mode', type=str, default='bilinear')
		parser.add_argument('--l1_loss_weight', type=float, default=0.1)
		parser.add_argument('--ssim_loss_weight', type=float, default=1.0)
		return parser

	def __init__(self, opt):
		super(BIGRESUNETModel, self).__init__(opt)

		self.opt = opt
		self.loss_names = [ 'UNET_L1', 'UNET_MSSIM', 'Total']
		self.visual_names = ['rainy_img', 'clean_img', 'derained_img']
		self.model_names = ['UNET']
		self.optimizer_names = ['UNET_optimizer_%s' % opt.optimizer]

		unet = UNET(
				ngf=opt.ngf,
				n_blocks=opt.n_blocks,
				norm_layer_type=opt.norm_layer_type,
				activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
				upsample_mode=opt.upsample_mode)
		self.netUNET = N.init_net(unet, opt.init_type, opt.init_gain, opt.gpu_ids)

		if self.isTrain:
			key_name_list = ['offset', 'modulator']
			deform_params = []
			normal_params = []
			for cur_name, parameters in self.netUNET.named_parameters():
				if any(key_name in cur_name for key_name in key_name_list):
					deform_params.append(parameters)
				else:
					normal_params.append(parameters)

			self.optimizer_UNET = optim.Adam(
				[{'params': normal_params},
				{'params': deform_params, 'lr': opt.lr / 10}],
				lr=opt.lr,
				betas=(0.9, 0.999),
				eps=1e-8)
			self.optimizers = [self.optimizer_UNET]

			self.criterionL1 = N.init_net(nn.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionMSSIM = N.init_net(L.ShiftMSSSIM(), gpu_ids=opt.gpu_ids)

	def set_input(self, input):
		self.rainy_img = input['rainy_img'].to(self.device)
		self.clean_img = input['clean_img'].to(self.device)
		self.name = input['file_name']

	def forward(self):
		self.derained_img = self.netUNET(self.rainy_img)

	def backward(self):
		self.loss_UNET_L1 = self.criterionL1(self.derained_img, self.clean_img).mean() ## 
		self.loss_UNET_MSSIM = self.criterionMSSIM(self.derained_img, self.clean_img).mean()
		# print(self.loss_UNET_L1.data)
		# print(self.loss_UNET_MSSIM.data)
		# exit(0)
		self.loss_Total = self.loss_UNET_L1 * self.opt.l1_loss_weight + \
							self.loss_UNET_MSSIM * self.opt.ssim_loss_weight
		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_UNET.zero_grad()
		self.backward()
		self.optimizer_UNET.step()

	def forward_x8(self):
		pass

	def update_before_iter(self):
		self.optimizer_UNET.zero_grad()
		self.optimizer_UNET.step()
		self.update_learning_rate()


class ResNetModified(nn.Module):
	"""
	Resnet-based generator that consists of deformable Resnet blocks.
	"""
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
		activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
		use_dropout=False, n_blocks=6, padding_type='reflect', upsample_mode='bilinear'):
		"""Construct a Resnet-based generator
		Parameters:
			input_nc (int) -- the number of channels in input images
			output_nc (int) -- the number of channels in output images
			ngf (int) -- the number of filters in the last conv layer
			norm_layer -- normalization layer
			use_dropout (bool) -- if use dropout layers
			n_blocks (int) -- the number of ResNet blocks
			padding_type (str) -- the name of padding layer in conv layers: reflect | replicate | zero
			upsample_mode (str) -- mode for upsampling: transpose | bilinear
		"""

		assert(n_blocks >= 0)
		
		super(ResNetModified, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		# Initial Convolution
		self.initial_conv = nn.Sequential(
			N.Conv2d(
				in_channels=input_nc,
				out_channels=ngf,
				kernel_size=7,
				padding_type=padding_type,
				norm_layer=norm_layer,
				activation_func=activation_func,
				use_bias=use_bias),
			N.Conv2d(
				in_channels=ngf,
				out_channels=ngf,
				kernel_size=3,
				padding_type=padding_type,
				norm_layer=norm_layer,
				activation_func=activation_func,
				use_bias=use_bias))

		# # Downsample Blocks
		# n_downsampling = 2
		mult = 2 ** 0

		# self.downsample_1 = N.Conv2d(
		# 	in_channels=ngf * mult,
		# 	out_channels=ngf * mult * 2,
		# 	kernel_size=3,
		# 	stride=2,
		# 	padding_type=padding_type,
		# 	norm_layer=norm_layer,
		# 	activation_func=activation_func,
		# 	use_bias=use_bias)

		# mult = 2 ** 1

		# self.downsample_2 = N.Conv2d(
		# 	in_channels=ngf * mult,
		# 	out_channels=ngf * mult * 2,
		# 	kernel_size=3,
		# 	stride=2,
		# 	padding_type=padding_type,
		# 	norm_layer=norm_layer,
		# 	activation_func=activation_func,
		# 	use_bias=use_bias)

		# Residual Blocks
		residual_blocks = []
		# mult = 2 ** n_downsampling

		for i in range(n_blocks): # add ResNet blocks
			residual_blocks += [
				N.DeformableResnetBlock(
					ngf * mult, 
					padding_type=padding_type, 
					norm_layer=norm_layer, 
					use_dropout=use_dropout, 
					use_bias=use_bias, activation_func=activation_func)]

		self.residual_blocks = nn.Sequential(*residual_blocks)

		# Upsampling
		# mult = 2 ** (n_downsampling - 0)

		# self.upsample_2 = N.DecoderBlock(
		# 	ngf * mult, 
		# 	int(ngf * mult / 2),
		# 	int(ngf * mult / 2),
		# 	use_bias=use_bias,
		# 	activation_func=activation_func,
		# 	norm_layer=norm_layer,
		# 	padding_type=padding_type,
		# 	upsample_mode=upsample_mode)

		# mult = 2 ** (n_downsampling - 1)

		# self.upsample_1 = N.DecoderBlock(
		# 	ngf * mult, 
		# 	int(ngf * mult / 2),
		# 	int(ngf * mult / 2),
		# 	use_bias=use_bias,
		# 	activation_func=activation_func,
		# 	norm_layer=norm_layer,
		# 	padding_type=padding_type,
		# 	upsample_mode=upsample_mode)

		# Output Convolution
		self.output_conv_naive = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0),
			nn.Tanh())

	def forward(self, input):
		"""Standard forward"""
		# Downsample
		initial_conv_out  = self.initial_conv(input)
		# downsample_1_out = self.downsample_1(initial_conv_out)
		# downsample_2_out = self.downsample_2(downsample_1_out)

		# Residual
		residual_blocks_out = self.residual_blocks(initial_conv_out)

		# Upsample
		# upsample_2_out = self.upsample_2(residual_blocks_out, downsample_1_out)
		# upsample_1_out = self.upsample_1(upsample_2_out, initial_conv_out)
		final_out = self.output_conv_naive(residual_blocks_out)

		# Return multiple final conv results
		return final_out

class UNET(nn.Module):   
	def __init__(self, ngf=64, n_blocks=9, norm_layer_type='batch',
		activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
		upsample_mode='bilinear'):
		"""
		GT-Rain Model
		Parameters:
			ngf (int) -- the number of conv filters
			n_blocks (int) -- the number of deformable ResNet blocks
			norm_layer_type (str) -- 'batch', 'instance'
			activation_func (func) -- activation functions
			upsample_mode (str) -- 'transpose', 'bilinear'
			init_type (str) -- None, 'normal', 'xavier', 'kaiming', 'orthogonal'
		"""
		super(UNET, self).__init__()

		self.resnet = ResNetModified(
			input_nc=3, output_nc=3, ngf=ngf, 
			norm_layer=N.get_norm_layer(norm_layer_type),
			activation_func=activation_func,
			use_dropout=False, n_blocks=n_blocks, 
			padding_type='reflect',
			upsample_mode=upsample_mode)

	def forward(self, x, res=True):
		out_img = self.resnet(x)
		if res:
			out = x + out_img
			out_img = torch.clip(out, -1, 1)
		return out_img
		