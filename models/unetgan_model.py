from email.policy import default
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch
import functools
from . import networks as N
from . import BaseModel as BaseModel
from . import losses as L
from skimage import exposure
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class UNETGANModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		parser.add_argument('--data_section', type=str, default='-1-1')
		parser.add_argument('--ngf', type=int, default=64)
		parser.add_argument('--n_blocks', type=int, default=9)
		parser.add_argument('--norm_layer_type', type=str, default='batch')
		parser.add_argument('--upsample_mode', type=str, default='bilinear')
		parser.add_argument('--l1_loss_weight', type=float, default=0.1)
		parser.add_argument('--ssim_loss_weight', type=float, default=1.0)
		parser.add_argument('--vgg19_loss_weight', type=float, default=0.0)
		parser.add_argument('--hist_matched_weight', type=float, default=0.0)
		parser.add_argument('--gan_loss_weight', type=float, default=0.1)
		parser.add_argument('--gan_type', type=str, default='vanilla')
		parser.add_argument('--test_internet', type=bool, default=False)
		return parser

	def __init__(self, opt):
		super(UNETGANModel, self).__init__(opt)

		self.opt = opt

		self.loss_names = ['Total_G', 'Total_D']
		if self.opt.l1_loss_weight > 0:
			self.loss_names.append('UNET_L1')
		if self.opt.ssim_loss_weight > 0:
			self.loss_names.append('UNET_MSSIM')
		if self.opt.gan_loss_weight > 0:
			self.loss_names.append('GAN')

		if self.opt.test_internet:
			self.visual_names = ['rainy_img', 'derained_img']
		else:
			self.visual_names = ['rainy_img', 'clean_img', 'derained_img']
		
		self.model_names = ['UNET', 'D']
		self.optimizer_names = ['UNET_optimizer_%s' % opt.optimizer, 'D_optimizer_%s' % opt.optimizer]

		unet = UNET(
				ngf=opt.ngf,
				n_blocks=opt.n_blocks,
				norm_layer_type=opt.norm_layer_type,
				activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
				upsample_mode=opt.upsample_mode)
		disnet = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True)

		self.netUNET = N.init_net(unet, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.netD = N.init_net(disnet, opt.init_type, opt.init_gain, opt.gpu_ids)

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
			
			self.optimizer_D = optim.Adam(   # 优化器参数还需要进行调整
				self.netD.parameters(),
				lr=opt.lr,
				betas=(0.9, 0.999),
				eps=1e-8)
			self.optimizers = [self.optimizer_UNET, self.optimizer_D]

			self.criterionL1 = N.init_net(nn.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionMSSIM = N.init_net(L.ShiftMSSSIM(), gpu_ids=opt.gpu_ids)
			if opt.vgg19_loss_weight > 0:
				self.critrionVGG19 = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
			if opt.gan_loss_weight > 0:
				self.criterionGAN = N.init_net(L.GANLoss(opt.gan_type), gpu_ids=opt.gpu_ids)

	def set_input(self, input):
		self.rainy_img = input['rainy_img'].to(self.device)
		if not self.opt.test_internet:
			self.clean_img = input['clean_img'].to(self.device)
		self.name = input['file_name']

	def forward(self):
		self.derained_img = self.netUNET(self.rainy_img)

	def backward_D(self):
		predict_fake = self.netD(self.derained_img.detach())
		loss_GAN_fake = self.criterionGAN(predict_fake, False, is_disc=True).mean()

		predict_real = self.netD(self.clean_img)
		loss_GAN_real = self.criterionGAN(predict_real, True, is_disc=True).mean()

		self.loss_Total_D = 0.5 * (loss_GAN_fake + loss_GAN_real)
		self.loss_Total_D.backward()
	
	def backward_G(self):
		self.loss_Total_G = 0

		if self.opt.ssim_loss_weight > 0:
			self.loss_UNET_MSSIM = self.criterionMSSIM(self.derained_img, self.clean_img).mean()
			self.loss_Total_G += self.opt.ssim_loss_weight * self.loss_UNET_MSSIM

		if self.opt.l1_loss_weight > 0:
			self.loss_UNET_L1 = self.criterionL1(self.derained_img, self.clean_img).mean() 
			self.loss_Total_G += self.opt.l1_loss_weight * self.loss_UNET_L1
		
		predict_fake = self.netD(self.derained_img)
		self.loss_GAN = self.criterionGAN(predict_fake, True, is_disc=False).mean()

		self.loss_Total_G += self.opt.gan_loss_weight * self.loss_GAN

		self.loss_Total_G.backward()

	def optimize_parameters(self):
		
		self.forward()

		# update D
		self.set_requires_grad(self.netD, True)
		self.optimizer_D.zero_grad()
		self.backward_D()
		self.optimizer_D.step()

		# update G
		self.set_requires_grad(self.netD, False)
		self.optimizer_UNET.zero_grad()
		self.backward_G()
		self.optimizer_UNET.step()

	def forward_x8(self):
		pass

	def update_before_iter(self):
		self.optimizer_UNET.zero_grad()
		self.optimizer_D.zero_grad()

		self.optimizer_UNET.step()
		self.optimizer_D.step()
		
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

		# Downsample Blocks
		n_downsampling = 2
		mult = 2 ** 0

		self.downsample_1 = N.Conv2d(
			in_channels=ngf * mult,
			out_channels=ngf * mult * 2,
			kernel_size=3,
			stride=2,
			padding_type=padding_type,
			norm_layer=norm_layer,
			activation_func=activation_func,
			use_bias=use_bias)

		mult = 2 ** 1

		self.downsample_2 = N.Conv2d(
			in_channels=ngf * mult,
			out_channels=ngf * mult * 2,
			kernel_size=3,
			stride=2,
			padding_type=padding_type,
			norm_layer=norm_layer,
			activation_func=activation_func,
			use_bias=use_bias)

		# Residual Blocks
		residual_blocks = []
		mult = 2 ** n_downsampling

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
		mult = 2 ** (n_downsampling - 0)

		self.upsample_2 = N.DecoderBlock(
			ngf * mult, 
			int(ngf * mult / 2),
			int(ngf * mult / 2),
			use_bias=use_bias,
			activation_func=activation_func,
			norm_layer=norm_layer,
			padding_type=padding_type,
			upsample_mode=upsample_mode)

		mult = 2 ** (n_downsampling - 1)

		self.upsample_1 = N.DecoderBlock(
			ngf * mult, 
			int(ngf * mult / 2),
			int(ngf * mult / 2),
			use_bias=use_bias,
			activation_func=activation_func,
			norm_layer=norm_layer,
			padding_type=padding_type,
			upsample_mode=upsample_mode)

		# Output Convolution
		self.output_conv_naive = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0),
			nn.Tanh())

	def forward(self, input):
		"""Standard forward"""
		# Downsample
		initial_conv_out  = self.initial_conv(input)
		downsample_1_out = self.downsample_1(initial_conv_out)
		downsample_2_out = self.downsample_2(downsample_1_out)

		# Residual
		residual_blocks_out = self.residual_blocks(downsample_2_out)

		# Upsample
		upsample_2_out = self.upsample_2(residual_blocks_out, downsample_1_out)
		upsample_1_out = self.upsample_1(upsample_2_out, initial_conv_out)
		final_out = self.output_conv_naive(upsample_1_out)

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

	def forward(self, x, res=False):
		out_img = self.resnet(x)
		if res:
			out_img += x
		return out_img

class UNetDiscriminatorSN(nn.Module):
	"""Defines a U-Net discriminator with spectral normalization (SN)
	It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
	Arg:
		num_in_ch (int): Channel number of inputs. Default: 3.
		num_feat (int): Channel number of base intermediate features. Default: 64.
		skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
	"""

	def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
		super(UNetDiscriminatorSN, self).__init__()
		self.skip_connection = skip_connection
		norm = spectral_norm
		# the first convolution
		self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
		# downsample
		self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
		self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
		self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
		# upsample
		self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
		self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
		self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
		# extra convolutions
		self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
		self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
		self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

	def forward(self, x):
		# downsample
		x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)  # H * W 
		x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
		x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
		x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

		# upsample
		x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
		x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

		if self.skip_connection:
			x4 = x4 + x2
		x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
		x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

		if self.skip_connection:
			x5 = x5 + x1
		x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
		x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

		if self.skip_connection:
			x6 = x6 + x0

		# extra convolutions
		out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
		out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
		out = self.conv9(out)

		return out

