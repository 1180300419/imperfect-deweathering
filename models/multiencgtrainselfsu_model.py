import torch.nn as nn
import torch.optim as optim
import torch
import functools
from . import networks as N
from . import BaseModel as BaseModel
from . import losses as L

class MULTIENCGTRAINSELFSUModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--data_section', type=str, default='-1-1')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--n_blocks', type=int, default=9)
        parser.add_argument('--norm_layer_type', type=str, default='batch')
        parser.add_argument('--upsample_mode', type=str, default='bilinear')
        parser.add_argument('--temperature', type=float, default=0.25)
        parser.add_argument('--l1_loss_weight', type=float, default=0.0)
        parser.add_argument('--ssim_loss_weight', type=float, default=0.0)
        parser.add_argument('--vgg19_loss_weight', type=float, default=0.0)
        parser.add_argument('--hist_matched_weight', type=float, default=0.0)
        parser.add_argument('--swd_loss_weight', type=float, default=0.1)
        parser.add_argument('--rain_variate_weight', type=float, default=0.1)
        parser.add_argument('--pseudo_l1_weight', type=float, default=1.0)
        parser.add_argument('--gradient_loss_weight', type=float, default=0.0)
        parser.add_argument('--laplacian_pyramid_weight', type=float, default=0.0)

        parser.add_argument('--test_internet', type=bool, default=False)
        return parser

    def __init__(self, opt):
        super(MULTIENCGTRAINSELFSUModel, self).__init__(opt)

        self.opt = opt

        self.loss_names = ['Total']
        if self.opt.l1_loss_weight > 0:
            self.loss_names.append('UNET_L1')
        if self.opt.ssim_loss_weight > 0:
            self.loss_names.append('UNET_MSSIM')
        if opt.rain_variate_weight > 0:
            self.loss_names.append('UNET_RAIN_VARIATE')

        if opt.vgg19_loss_weight > 0:
            self.loss_names.append('UNET_VGG19')
        if opt.hist_matched_weight > 0:
            self.loss_names.append('UNET_HISTED')
        if opt.gradient_loss_weight > 0:
            self.loss_names.append('UNET_GRADIENT')
        if opt.laplacian_pyramid_weight > 0:
            self.loss_names.append('UNET_LAPLACIAN')
        if self.opt.pseudo_l1_weight > 0:
            self.loss_names.append('UNETSUPER_L1')
        if self.opt.swd_loss_weight > 0:
            self.loss_names.append('UNETSUPER_SWD')

        if self.opt.test_internet:
            self.visual_names = ['rainy_img', 'derained_img']
        else:
            self.visual_names = ['rainy_img', 'clean_img', 'derained_img', 'single_rainy_img']

        self.model_names = ['UNET']
        self.optimizer_names = ['UNET_optimizer_%s' % opt.optimizer]

        unet = UNET(
                ngf=opt.ngf,
                n_blocks=opt.n_blocks,
                norm_layer_type=opt.norm_layer_type,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                upsample_mode=opt.upsample_mode)
        self.netUNET = N.init_net(unet, opt.init_type, opt.init_gain, opt.gpu_ids)

        unet_supervise = UNET1(
                ngf=opt.ngf,
                n_blocks=opt.n_blocks,
                norm_layer_type=opt.norm_layer_type,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                upsample_mode=opt.upsample_mode)
        self.netUNETSUPERVISE = N.init_net(unet_supervise, opt.init_type, opt.init_gain, opt.gpu_ids)
        
        # self.load_network_path(self.netUNETSUPERVISE, '/hdd1/lxh/derain/code/checkpoints/teacher_gtrain-rain-snow_dataall/UNET_model_12.pth')       
        
        self.set_requires_grad(self.netUNETSUPERVISE, False)
        self.netUNETSUPERVISE.eval()

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

            if self.opt.l1_loss_weight > 1e-6:
                self.criterionL1 = N.init_net(nn.L1Loss(), gpu_ids=opt.gpu_ids)
            if self.opt.pseudo_l1_weight > 1e-6:
                self.criterion_pseudol1 = N.init_net(nn.L1Loss(), gpu_ids=opt.gpu_ids)
            if self.opt.ssim_loss_weight > 1e-6:
                self.criterionMSSIM = N.init_net(L.ShiftMSSSIM(), gpu_ids=opt.gpu_ids)
            if opt.rain_variate_weight > 1e-6:
                self.criterionRainVarient = N.init_net(L.RainRobustLoss(batch_size=opt.batch_size // len(opt.gpu_ids),
                                                                        n_views=2,
                                                                        temperature=opt.temperature), gpu_ids=opt.gpu_ids)

            if opt.vgg19_loss_weight > 1e-6:
                self.critrionVGG19 = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)

            if opt.gradient_loss_weight > 1e-6:
                self.criterionGradient = N.init_net(L.GWLoss(w=4, reduction='mean'), gpu_ids=opt.gpu_ids)
            if opt.laplacian_pyramid_weight > 1e-6:
                self.criterionLaplacian = N.init_net(L.LapPyrLoss(num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean'), gpu_ids=opt.gpu_ids)
            if self.opt.swd_loss_weight > 0:
                self.criterionSWD = N.init_net(L.SWDLoss(), gpu_ids=opt.gpu_ids)
            
    def set_input(self, input, epoch):
        if self.isTrain:
            self.rainy_img = input['single_rainy_img'].to(self.device)
            self.rainyT_img = torch.cat(input['rainy_img'], dim=1).to(self.device)
            self.clean_img = input['clean_img'].to(self.device)
            self.name = input['file_name']
        else:
            self.rainy_img = input['single_rainy_img'].to(self.device)
            self.single_rainy_img = input['single_rainy_img'].to(self.device)
            if not self.opt.test_internet:
                self.folder = input['folder']
                self.clean_img = input['clean_img'].to(self.device)
            self.name = input['file_name']

        
    def forward(self):
        if self.isTrain:
            self.derained_img, self.features = self.netUNET(self.rainy_img, self.clean_img)
            if self.opt.pseudo_l1_weight > 0:
                with torch.no_grad():
                    self.supervise = self.netUNETSUPERVISE(self.rainyT_img)
        else:
            self.derained_img, _ = self.netUNET(self.rainy_img)

    def backward(self, epoch):

        self.loss_Total = 0

        if self.opt.ssim_loss_weight > 0:
            self.loss_UNET_MSSIM = self.criterionMSSIM(self.derained_img, self.clean_img).mean()
            self.loss_Total += self.opt.ssim_loss_weight * self.loss_UNET_MSSIM

        if self.opt.l1_loss_weight > 0:
            self.loss_UNET_L1 = self.criterionL1(self.derained_img, self.clean_img).mean() ## 
            self.loss_Total += self.opt.l1_loss_weight * self.loss_UNET_L1

        if self.opt.rain_variate_weight > 0:
            b = self.features.shape[0] // 2
            self.loss_UNET_RAIN_VARIATE = self.criterionRainVarient(self.features[:b, ...], self.features[b:, ...], self.name).mean()
            self.loss_Total += self.opt.rain_variate_weight * self.loss_UNET_RAIN_VARIATE
        
        if self.opt.pseudo_l1_weight > 0:
            self.loss_UNETSUPER_L1 = self.criterion_pseudol1(self.derained_img, self.supervise).mean()
            self.loss_Total += self.opt.pseudo_l1_weight * self.loss_UNETSUPER_L1
        
        if self.opt.swd_loss_weight > 0:
            self.loss_UNETSUPER_SWD = self.criterionSWD(self.derained_img, self.clean_img, 3).mean()
            self.loss_Total += self.opt.swd_loss_weight * self.loss_UNETSUPER_SWD

        self.loss_Total.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer_UNET.zero_grad()
        self.backward(epoch)
        torch.nn.utils.clip_grad_norm_(self.netUNET.parameters(), 0.1)
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

        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(start_dim=1, end_dim=-1))

    def forward(self, input):
        """Standard forward"""
        # Downsample
        # import pdb; pdb.set_trace()
        initial_conv_out  = self.initial_conv(input)
        downsample_1_out = self.downsample_1(initial_conv_out)
        downsample_2_out = self.downsample_2(downsample_1_out)

        # Residual
        residual_blocks_out = self.residual_blocks(downsample_2_out)

        # Upsample
        upsample_2_out = self.upsample_2(residual_blocks_out, downsample_1_out)
        upsample_1_out = self.upsample_1(upsample_2_out, initial_conv_out)
        final_out = self.output_conv_naive(upsample_1_out)

        features = self.feature_projection(residual_blocks_out)
        # Return multiple final conv results
        return final_out, features

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

    def forward(self, x, clean_img=None):
        if clean_img is None:
            out_img, out_feature = self.resnet(x)
        else:
            input_cat = torch.cat((x, clean_img), dim=0)
            out_img, out_feature = self.resnet(input_cat)
        out_img = torch.clip(out_img, -1, 1)
        return out_img[:x.shape[0], ...], out_feature

class Encoder(nn.Module):
    """
    Resnet-based generator that consists of deformable Resnet blocks.
    """
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
        activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True), 
        n_blocks=6,
        use_dropout=False,
        padding_type='reflect'):
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

        super(Encoder, self).__init__()

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
        
    def forward(self, input):
        """Standard forward"""
        initial_conv_out  = self.initial_conv(input)
        downsample_1_out = self.downsample_1(initial_conv_out)
        downsample_2_out = self.downsample_2(downsample_1_out)
        resblocks_out = self.residual_blocks(downsample_2_out)
        return initial_conv_out, downsample_1_out, resblocks_out
    
class ResNetModified1(nn.Module):
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

        super(ResNetModified1, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.encoder1 = Encoder(3)
        self.encoder2 = Encoder(3)
        self.encoder3 = Encoder(3)
        self.encoder4 = Encoder(3)
        self.encoder5 = Encoder(3)

        n_encoder = 5
        self.modify_channel = N.ModifyChannelWCA(256 * n_encoder, 256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=True, activation_func=activation_func)
        self.modify_channel_initialconv = N.ModifyChannelWCA(ngf * n_encoder, ngf, padding_type, norm_layer, False, True, activation_func)
        self.modify_channel_down1 = N.ModifyChannelWCA(ngf * n_encoder * 2, ngf * 2, padding_type, norm_layer, False, True, activation_func)

        # Downsample Blocks
        n_downsampling = 2
        mult = 2 ** n_downsampling


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
        initial_conv_out1, downsample_1_out1, residual_blocks_out1 = self.encoder1(input[:, 0:3, ...])
        initial_conv_out2, downsample_1_out2, residual_blocks_out2 = self.encoder2(input[:, 3:6, ...])
        initial_conv_out3, downsample_1_out3, residual_blocks_out3 = self.encoder3(input[:, 6:9, ...])
        initial_conv_out4, downsample_1_out4, residual_blocks_out4 = self.encoder4(input[:, 9:12, ...])
        initial_conv_out5, downsample_1_out5, residual_blocks_out5 = self.encoder5(input[:, 12:15, ...])

        
        residual_blocks_out = self.modify_channel(torch.cat([residual_blocks_out1, residual_blocks_out2, 
                                residual_blocks_out3, residual_blocks_out4, residual_blocks_out5], dim=1))
        
        downsample_1_out = self.modify_channel_down1(torch.cat([downsample_1_out1, downsample_1_out2, 
                                downsample_1_out3, downsample_1_out4, downsample_1_out5], dim=1))
        initial_conv_out = self.modify_channel_initialconv(torch.cat([initial_conv_out1, initial_conv_out2, initial_conv_out3,
                                initial_conv_out4, initial_conv_out5], dim=1))
        
        # Upsample
        upsample_2_out = self.upsample_2(residual_blocks_out, downsample_1_out)
        upsample_1_out = self.upsample_1(upsample_2_out, initial_conv_out)
        final_out = self.output_conv_naive(upsample_1_out)

        # Return multiple final conv results
        return final_out

class UNET1(nn.Module):   
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
        super(UNET1, self).__init__()

        self.resnet = ResNetModified1(
            input_nc=3, output_nc=3, ngf=ngf, 
            norm_layer=N.get_norm_layer(norm_layer_type),
            activation_func=activation_func,
            use_dropout=False, n_blocks=n_blocks, 
            padding_type='reflect',
            upsample_mode=upsample_mode)

    def forward(self, x):
        out_img = self.resnet(x)
        out_img = torch.clip(out_img, -1, 1)
        return out_img
      

