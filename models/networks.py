import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module
import functools
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
import torchvision.ops as ops
import torchvision

"""
Linear warmup from: 
https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
"""

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
  """ Gradually warm-up(increasing) learning rate in optimizer.
  Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
  Args:
      optimizer (Optimizer): Wrapped optimizer.
      multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
      total_epoch: target learning rate is reached at total_epoch, gradually
      after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
  """

  def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
    self.multiplier = multiplier
    if self.multiplier < 1.:
      raise ValueError('multiplier should be greater thant or equal to 1.')
    self.total_epoch = total_epoch
    self.after_scheduler = after_scheduler
    self.finished = False
    super(GradualWarmupScheduler, self).__init__(optimizer)

  def get_lr(self):
    if self.last_epoch > self.total_epoch:
      if self.after_scheduler:
        if not self.finished:
          self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
          self.finished = True
        return self.after_scheduler.get_last_lr()
      return [base_lr * self.multiplier for base_lr in self.base_lrs]

    if self.multiplier == 1.0:
      return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
    else:
      return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

  def step_ReduceLROnPlateau(self, metrics, epoch=None):
    if epoch is None:
      epoch = self.last_epoch + 1
    self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
    if self.last_epoch <= self.total_epoch:
      warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
      for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
        param_group['lr'] = lr
    else:
      if epoch is None:
        self.after_scheduler.step(metrics, None)
      else:
        self.after_scheduler.step(metrics, epoch - self.total_epoch)

  def step(self, epoch=None, metrics=None):
    if type(self.after_scheduler) != ReduceLROnPlateau:
      if self.finished and self.after_scheduler:
        if epoch is None:
          self.after_scheduler.step(None)
        else:
          self.after_scheduler.step(epoch - self.total_epoch)
        self._last_lr = self.after_scheduler.get_last_lr()
      else:
        return super(GradualWarmupScheduler, self).step(epoch)
    else:
      self.step_ReduceLROnPlateau(metrics, epoch)

def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,
						step_size=opt.lr_decay_iters,
						gamma=0.5)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
								mode='min',
								factor=0.2,
								threshold=0.01,
								patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
								T_max=opt.niter,
								eta_min=0)
					
	elif opt.lr_policy == 'warmup':
		scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
			optimizer, 
			opt.niter - opt.warmup_niter, 
			eta_min=opt.min_lr)

		scheduler = GradualWarmupScheduler(
			optimizer, 
			multiplier=1.0, 
			total_epoch=opt.warmup_niter, 
			after_scheduler=scheduler_cosine)
	return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
				or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			elif init_type == 'uniform':
				init.uniform_(m.weight.data, b=init_gain)
			else:
				raise NotImplementedError('[%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	if init_type != 'default' and init_type is not None:
		init_weights(net, init_type, init_gain=init_gain)
	return net

'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
	if len(args) == 1:
		args = args[0]
	if isinstance(args, nn.Module):
		return args
	modules = OrderedDict()
	if isinstance(args, OrderedDict):
		for k, v in args.items():
			modules[k] = seq(v)
		return nn.Sequential(modules)
	assert isinstance(args, (list, tuple))
	return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# ===================================
'''

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
		 output_padding=0, dilation=1, groups=1, bias=True,
		 padding_mode='zeros', mode='CBR'):
	L = []
	for t in mode:
		if t == 'C':
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=groups,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'X':
			assert in_channels == out_channels
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=in_channels,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'T':
			L.append(nn.ConvTranspose2d(in_channels=in_channels,
										out_channels=out_channels,
										kernel_size=kernel_size,
										stride=stride,
										padding=padding,
										output_padding=output_padding,
										groups=groups,
										bias=bias,
										dilation=dilation,
										padding_mode=padding_mode))
		elif t == 'B':
			L.append(nn.BatchNorm2d(out_channels))
		elif t == 'I':
			L.append(nn.InstanceNorm2d(out_channels, affine=True))
		elif t == 'i':
			L.append(nn.InstanceNorm2d(out_channels))
		elif t == 'R':
			L.append(nn.ReLU(inplace=True))
		elif t == 'r':
			L.append(nn.ReLU(inplace=False))
		elif t == 'S':
			L.append(nn.Sigmoid())
		elif t == 'P':
			L.append(nn.PReLU())
		elif t == 'L':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
		elif t == 'l':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
		elif t == '2':
			L.append(nn.PixelShuffle(upscale_factor=2))
		elif t == '3':
			L.append(nn.PixelShuffle(upscale_factor=3))
		elif t == '4':
			L.append(nn.PixelShuffle(upscale_factor=4))
		elif t == 'U':
			L.append(nn.Upsample(scale_factor=2, mode='nearest'))
		elif t == 'u':
			L.append(nn.Upsample(scale_factor=3, mode='nearest'))
		elif t == 'M':
			L.append(nn.MaxPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		elif t == 'A':
			L.append(nn.AvgPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return seq(*L)


# Main network blocks
# Code modified from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# Basic Blocks
class Identity(nn.Module):
  def forward(self, x):
    return x

def get_norm_layer(norm_type='instance'):
  """Return a normalization layer
  Parameters:
      norm_type (str) -- the name of the normalization layer: batch | instance | none
  For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
  For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
  """
  if norm_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
  elif norm_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
  elif norm_type == 'none':
    def norm_layer(x): return Identity()
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
  return norm_layer

class Conv2d(torch.nn.Module):
  '''
  2D convolution class
  Args:
    in_channels : int
      number of input channels
    out_channels : int
      number of output channels
    kernel_size : int
      size of kernel
    stride : int
      stride of convolution
    activation_func : func
      activation function after convolution
    norm_layer : functools.partial
      normalization layer
    use_bias : bool
      if set, then use bias
    padding_type : str
      the name of padding layer: reflect | replicate | zero
  '''

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size=3,
      stride=1,
      activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
      norm_layer=nn.BatchNorm2d,
      use_bias=False,
      padding_type='reflect'):
    
    super(Conv2d, self).__init__()
    
    self.activation_func = activation_func
    conv_block = []
    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(kernel_size // 2)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(kernel_size // 2)]
    elif padding_type == 'zero':
      p = kernel_size // 2
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [
      nn.Conv2d(
          in_channels, 
          out_channels, 
          stride=stride,
          kernel_size=kernel_size, 
          padding=p, 
          bias=use_bias), 
      norm_layer(out_channels)]

    self.conv = nn.Sequential(*conv_block)

  def forward(self, x):
    conv = self.conv(x)

    if self.activation_func is not None:
      return self.activation_func(conv)
    else:
      return conv

class DeformableConv2d(nn.Module):
  '''
  2D deformable convolution class
  Args:
    in_channels : int
      number of input channels
    out_channels : int
      number of output channels
    kernel_size : int
      size of kernel
    stride : int
      stride of convolution
    padding : int
      padding
    use_bias : bool
      if set, then use bias
  '''
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False):
    
    super(DeformableConv2d, self).__init__()
      
    self.stride = stride if type(stride) == tuple else (stride, stride)
    self.padding = padding
    
    self.offset_conv = nn.Conv2d(
        in_channels, 
        2 * kernel_size * kernel_size,
        kernel_size=kernel_size, 
        stride=stride,
        padding=self.padding, 
        bias=True)

    nn.init.constant_(self.offset_conv.weight, 0.)
    nn.init.constant_(self.offset_conv.bias, 0.)
    
    self.modulator_conv = nn.Conv2d(
        in_channels, 
        1 * kernel_size * kernel_size,
        kernel_size=kernel_size, 
        stride=stride,
        padding=self.padding, 
        bias=True)

    nn.init.constant_(self.modulator_conv.weight, 0.)
    nn.init.constant_(self.modulator_conv.bias, 0.)
    
    self.regular_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=self.padding,
        bias=bias)

  def forward(self, x):
    offset = self.offset_conv(x)
    modulator = 2. * torch.sigmoid(self.modulator_conv(x))
    
    x = torchvision.ops.deform_conv2d(
        input=x, 
        offset=offset, 
        weight=self.regular_conv.weight, 
        bias=self.regular_conv.bias, 
        padding=self.padding,
        mask=modulator,
        stride=self.stride)

    return x

class UpConv2d(torch.nn.Module):
  '''
  Up-convolution (upsample + convolution) block class
  Args:
    in_channels : int
      number of input channels
    out_channels : int
      number of output channels
    kernel_size : int
      size of kernel (k x k)
    activation_func : func
      activation function after convolution
    norm_layer : functools.partial
      normalization layer
    use_bias : bool
      if set, then use bias
    padding_type : str
      the name of padding layer: reflect | replicate | zero
    interpolate_mode : str
      the mode for interpolation: bilinear | nearest
  '''
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size=3,
      activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
      norm_layer=nn.BatchNorm2d,
      use_bias=False,
      padding_type='reflect',
      interpolate_mode='bilinear'):
    
    super(UpConv2d, self).__init__()
    self.interpolate_mode = interpolate_mode

    self.conv = Conv2d(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=1,
      activation_func=activation_func,
      norm_layer=norm_layer,
      use_bias=use_bias,
      padding_type=padding_type)

  def forward(self, x):
    n_height, n_width = x.shape[2:4]
    shape = (int(2 * n_height), int(2 * n_width))
    upsample = torch.nn.functional.interpolate(
      x, size=shape, mode=self.interpolate_mode, align_corners=True)
    conv = self.conv(upsample)
    return conv

class DeformableResnetBlock(nn.Module):
  """Define a Resnet block with deformable convolutions"""
  
  def __init__(
    self, dim, padding_type, 
    norm_layer, use_dropout, 
    use_bias, activation_func):
  
    """Initialize the deformable Resnet block
    A defromable resnet block is a conv block with skip connections
    """
    super(DeformableResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(
        dim, padding_type, 
        norm_layer, use_dropout, 
        use_bias, activation_func)

  def build_conv_block(
    self, dim, padding_type, 
    norm_layer, use_dropout, 
    use_bias, activation_func):
    """Construct a convolutional block.
    Parameters:
        dim (int) -- the number of channels in the conv layer.
        padding_type (str) -- the name of padding layer: reflect | replicate | zero
        norm_layer -- normalization layer
        use_dropout (bool) -- if use dropout layers.
        use_bias (bool) -- if the conv layer uses bias or not
        activation_func (func) -- activation type
    Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
    """
    conv_block = []

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [
        DeformableConv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
        norm_layer(dim), 
        activation_func]
    
    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    conv_block += [DeformableConv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    """Forward function (with skip connections)"""
    out = x + self.conv_block(x)    # add skip connections
    return out

# 将DeformableResnetBlock中的可形变卷积操作替换成普通卷积
class ResnetBlock(nn.Module):
  """Define a Resnet block with normal convolutions"""
  
  def __init__(
    self, dim, padding_type, 
    norm_layer, use_dropout, 
    use_bias, activation_func):
  
    """Initialize the Resnet block
    A  resnet block is a conv block with skip connections
    """
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(
        dim, padding_type, 
        norm_layer, use_dropout, 
        use_bias, activation_func)

  def build_conv_block(
    self, dim, padding_type, norm_layer, use_dropout, use_bias, activation_func):
    """Construct a convolutional block.
    Parameters:
        dim (int) -- the number of channels in the conv layer.
        padding_type (str) -- the name of padding layer: reflect | replicate | zero
        norm_layer -- normalization layer
        use_dropout (bool) -- if use dropout layers.
        use_bias (bool) -- if the conv layer uses bias or not
        activation_func (func) -- activation type
    Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
    """

    conv_block = []

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [
        nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
        norm_layer(dim), 
        activation_func]
    
    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    """Forward function (with skip connections)"""
    out = x + self.conv_block(x)    # add skip connections
    return out

class DecoderBlock(torch.nn.Module):
  '''
  Decoder block with skip connections
  Args:
    in_channels : int
      number of input channels
    skip_channels : int
      number of skip connection channels
    out_channels : int
      number of output channels
    activation_func : func
      activation function after convolution
    norm_layer : functools.partial
      normalization layer
    use_bias : bool
      if set, then use bias
    padding_type : str
      the name of padding layer: reflect | replicate | zero
    upsample_mode : str
      the mode for interpolation: transpose | bilinear | nearest
  '''

  def __init__(
      self,
      in_channels,
      skip_channels,
      out_channels,
      activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
      norm_layer=nn.BatchNorm2d,
      use_bias=False,
      padding_type='reflect',
      upsample_mode='transpose'):
    super(DecoderBlock, self).__init__()

    self.skip_channels = skip_channels
    self.upsample_mode = upsample_mode
    
    # Upsampling
    if upsample_mode == 'transpose':
      self.deconv = nn.Sequential(
          nn.ConvTranspose2d(
              in_channels, out_channels,
              kernel_size=3, stride=2,
              padding=1, output_padding=1,
              bias=use_bias),
          norm_layer(out_channels),
          activation_func)
    else:
      self.deconv = UpConv2d(
          in_channels, out_channels,
          use_bias=use_bias,
          activation_func=activation_func,
          norm_layer=norm_layer,
          padding_type=padding_type,
          interpolate_mode=upsample_mode)

    concat_channels = skip_channels + out_channels
    
    self.conv = Conv2d(
        concat_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation_func=activation_func,
        padding_type=padding_type,
        norm_layer=norm_layer,
        use_bias=use_bias)

  def forward(self, x, skip=None):
    deconv = self.deconv(x)

    if self.skip_channels > 0:
      concat = torch.cat([deconv, skip], dim=1)
    else:
      concat = deconv

    return self.conv(concat)

class DecoderBlockIWT(torch.nn.Module):
  '''
  Decoder block with skip connections
  Args:
    in_channels : int
      number of input channels
    skip_channels : int
      number of skip connection channels
    out_channels : int
      number of output channels
    activation_func : func
      activation function after convolution
    norm_layer : functools.partial
      normalization layer
    use_bias : bool
      if set, then use bias
    padding_type : str
      the name of padding layer: reflect | replicate | zero
    upsample_mode : str
      the mode for interpolation: transpose | bilinear | nearest
  '''

  def __init__(
      self,
      in_channels,
      skip_channels,
      out_channels,
      activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
      norm_layer=nn.BatchNorm2d,
      use_bias=False,
      padding_type='reflect',
      upsample_mode='transpose'):
    super(DecoderBlockIWT, self).__init__()

    self.skip_channels = skip_channels
    # self.upsample_mode = upsample_mode
    
    # Upsampling
    
    self.deconv = Conv2d(
      in_channels,
      out_channels,
      kernel_size=3,
      stride=1,
      activation_func=activation_func,
      norm_layer=norm_layer,
      use_bias=use_bias,
      padding_type=padding_type)

    concat_channels = skip_channels + out_channels
    
    self.conv = Conv2d(
        concat_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation_func=activation_func,
        padding_type=padding_type,
        norm_layer=norm_layer,
        use_bias=use_bias)

  def forward(self, x, skip=None):
    deconv = self.deconv(x)

    if self.skip_channels > 0:
      concat = torch.cat([deconv, skip], dim=1)
    else:
      concat = deconv

    return self.conv(concat)

def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(torch.tensor(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
        # exit(0)
        # mean_x
        mean_x = self.boxfilter(x) / N
        # exit(0)
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        # exit(0)
        mean_A = self.boxfilter(A) / N
        # exit(0)
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b

# 将GT和Derained同时输入GCMModel
class GCMModel(nn.Module):
    def __init__(self):
        super(GCMModel, self).__init__()
        self.ch_1 = 32
        self.ch_2 = 64
    
        guide_input_channels = 6
        align_input_channels = 3

        self.guide_net = seq(
            conv(guide_input_channels, self.ch_1, 7, stride=2, padding=0, mode='CR'),
            conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRC'),
            nn.AdaptiveAvgPool2d(1),
            conv(self.ch_1, self.ch_2, 1, stride=1, padding=0, mode='C')
        )

        self.align_head = conv(align_input_channels, self.ch_2, 1, padding=0, mode='CR')

        self.align_base = seq(
            conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0, mode='CRCRCR')
        )
        self.align_tail = seq(
            conv(self.ch_2, 3, 1, padding=0, mode='C')
        )

    def forward(self, demosaic_raw, dslr):
        # convert the color of demosaic_raw to dslr
        # demosaic_raw = torch.pow(demosaic_raw, 1/2.2)
        guide_input = torch.cat((demosaic_raw, dslr), 1)
        base_input = demosaic_raw

        guide = self.guide_net(guide_input)

        out = self.align_head(base_input)
        out = guide * out + out
        out = self.align_base(out)
        out = self.align_tail(out) + demosaic_raw

        return out

# 仅将derained输入GCMModel
class GCMModelOnlyX(nn.Module):
    def __init__(self):
        super(GCMModelOnlyX, self).__init__()
        self.ch_1 = 32
        self.ch_2 = 64
    
        guide_input_channels = 3
        align_input_channels = 3

        self.guide_net = seq(
            conv(guide_input_channels, self.ch_1, 7, stride=2, padding=0, mode='CR'),
            conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRC'),
            nn.AdaptiveAvgPool2d(1),
            conv(self.ch_1, self.ch_2, 1, stride=1, padding=0, mode='C')
        )

        self.align_head = conv(align_input_channels, self.ch_2, 1, padding=0, mode='CR')

        self.align_base = seq(
            conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0, mode='CRCRCR')
        )
        self.align_tail = seq(
            conv(self.ch_2, 3, 1, padding=0, mode='C')
        )

    def forward(self, demosaic_raw, dslr=None):
        # convert the color of demosaic_raw to dslr
        # demosaic_raw = torch.pow(demosaic_raw, 1/2.2)
        # guide_input = torch.cat((demosaic_raw, dslr), 1)
        guide_input = demosaic_raw
        base_input = demosaic_raw

        guide = self.guide_net(guide_input)

        out = self.align_head(base_input)
        out = guide * out + out
        out = self.align_base(out)
        out = self.align_tail(out) + demosaic_raw

        return out


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)+dilation-1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):
    return nn.Conv2d(
        in_channels,out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups=groups)

#def shuffle_channel()

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def pixel_down_shuffle(x, downsacale_factor):
    batchsize, num_channels, height, width = x.size()

    out_height = height // downsacale_factor
    out_width = width // downsacale_factor
    input_view = x.contiguous().view(batchsize, num_channels, out_height, downsacale_factor, out_width,
                                     downsacale_factor)

    num_channels *= downsacale_factor ** 2
    unshuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()

    return unshuffle_out.view(batchsize, num_channels, out_height, out_width)

def sp_init(x):

    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x_LL = x01[:, :, :, 0::2]
    x_HL = x02[:, :, :, 0::2]
    x_LH = x01[:, :, :, 1::2]
    x_HH = x02[:, :, :, 1::2]


    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class Channel_Shuffle(nn.Module):
    def __init__(self, conv_groups):
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        return channel_shuffle(x, self.conv_groups)

class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return sp_init(x)

class Pixel_Down_Shuffle(nn.Module):
    def __init__(self):
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return pixel_down_shuffle(x, 2)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.create_graph = False
            self.volatile = True
class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift2, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1)
        self.weight.data.div_(std.view(4, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.volatile = True

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=False, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class BBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x

class DBlock_com(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class ShuffleBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,conv_groups=1):

        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):
    def __init__(
        self, conv, conv1, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
