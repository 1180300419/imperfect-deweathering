import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseModel as BaseModel
from skimage import exposure
from util.util import rgbten2ycbcrten
from . import networks as N
import torch.optim as optim
from . import losses as L


class REALUNETModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--data_section', type=str, default='-1-1')

        parser.add_argument('--l1_loss_weight', type=float, default=0.1)
        parser.add_argument('--ssim_loss_weight', type=float, default=1.0)
        parser.add_argument('--vgg19_loss_weight', type=float, default=0.0)
        parser.add_argument('--hist_matched_weight', type=float, default=0.0)

        parser.add_argument('--gradient_loss_weight', type=float, default=0.0)
        parser.add_argument('--laplacian_pyramid_weight', type=float, default=0.0)

        parser.add_argument('--test_internet', type=bool, default=False)
        return parser

    def __init__(self, opt):
        super(REALUNETModel, self).__init__(opt)
        self.opt = opt

        self.loss_names = ['Total']
        if self.opt.l1_loss_weight > 0:
            self.loss_names.append('REALUNET_L1')
        if self.opt.ssim_loss_weight > 0:
            self.loss_names.append('REALUNET_MSSIM')
        if opt.vgg19_loss_weight > 0:
            self.loss_names.append('REALUNET_VGG19')
        if opt.hist_matched_weight > 0:
            self.loss_names.append('REALUNET_HISTED')
        if opt.gradient_loss_weight > 0:
            self.loss_names.append('REALUNET_GRADIENT')
        if opt.laplacian_pyramid_weight > 0:
            self.loss_names.append('REALUNET_LAPLACIAN')

        if self.opt.test_internet:
            self.visual_names = ['rainy_img', 'derained_img']
        else:
            self.visual_names = ['rainy_img', 'clean_img', 'derained_img']
        self.model_names = ['REALUNET']
        self.optimizer_names = ['REALUNET_optimizer_%s' % opt.optimizer]

        realunet = REALUNet(in_channels=3, out_channels=3)
        self.netREALUNET = N.init_net(realunet, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:

            self.optimizer_REALUNET = optim.Adam(
                self.netREALUNET.parameters(),
                lr=opt.lr,
                betas=(0.9, 0.999),
                eps=1e-8)
            self.optimizers = [self.optimizer_REALUNET]

            if self.opt.l1_loss_weight > 0:
                self.criterionL1 = N.init_net(nn.L1Loss(), gpu_ids=opt.gpu_ids)
            if self.opt.ssim_loss_weight > 0:
                self.criterionMSSIM = N.init_net(L.ShiftMSSSIM(), gpu_ids=opt.gpu_ids)
            if opt.vgg19_loss_weight > 0:
                self.critrionVGG19 = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)

            if opt.gradient_loss_weight > 0:
                self.criterionGradient = N.init_net(L.GWLoss(w=4, reduction='mean'), gpu_ids=opt.gpu_ids)
            if opt.laplacian_pyramid_weight > 0:
                self.criterionLaplacian = N.init_net(L.LapPyrLoss(num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean'), gpu_ids=opt.gpu_ids)

    def set_input(self, input):
        self.rainy_img = input['rainy_img'].to(self.device)
        if not self.opt.test_internet:
            self.clean_img = input['clean_img'].to(self.device)
        self.name = input['file_name']

    def forward(self):
        self.derained_img = self.netREALUNET(self.rainy_img)

    def backward(self):

        self.loss_Total = 0

        if self.opt.ssim_loss_weight > 0:
            self.loss_REALUNET_MSSIM = self.criterionMSSIM(self.derained_img, self.clean_img).mean()
            self.loss_Total += self.opt.ssim_loss_weight * self.loss_REALUNET_MSSIM

        if self.opt.l1_loss_weight > 0:
            self.loss_REALUNET_L1 = self.criterionL1(self.derained_img, self.clean_img).mean() ## 
            self.loss_Total += self.opt.l1_loss_weight * self.loss_REALUNET_L1

        if self.opt.vgg19_loss_weight > 0:
            self.loss_REALUNET_VGG19 = self.critrionVGG19(self.derained_img, self.clean_img).mean()
            self.loss_Total += self.opt.vgg19_loss_weight * self.loss_REALUNET_VGG19

        if self.opt.hist_matched_weight > 0:
            for m in range(self.derained_img.shape[0]):
                derained = self.derained_img[m].detach().cpu().numpy()
                clean = self.clean_img[m].detach().cpu().numpy()
                img_np = exposure.match_histograms(clean, derained, multichannel=True)
                self.clean_img[m] = torch.from_numpy(img_np).to(self.device)
                
            self.loss_REALUNET_HISTED = self.criterionL1(self.derained_img, self.clean_img).mean()
            self.loss_Total += self.opt.hist_matched_weight * self.loss_REALUNET_HISTED
        
        if self.opt.gradient_loss_weight > 0 or self.opt.laplacian_pyramid_weight > 0:
            derained_ycbcr = rgbten2ycbcrten(self.derained_img, only_y=False)
            clean_ycbcr = rgbten2ycbcrten(self.clean_img, only_y=False)

        if self.opt.laplacian_pyramid_weight > 0:
            self.loss_REALUNET_LAPLACIAN = self.criterionLaplacian(derained_ycbcr[:, :1, ...], clean_ycbcr[:, :1, ...]).mean()
            self.loss_Total += self.opt.laplacian_pyramid_weight * self.loss_REALUNET_LAPLACIAN
        if self.opt.gradient_loss_weight > 0:
            self.loss_REALUNET_GRADIENT = self.criterionGradient(derained_ycbcr[:, 1:, ...], clean_ycbcr[:, 1:, ...]).mean()
            self.loss_Total += self.opt.gradient_loss_weight * self.loss_REALUNET_GRADIENT

        self.loss_Total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_REALUNET.zero_grad()
        self.backward()
        torch.nn.utils.clip_grad_norm_(self.netREALUNET.parameters(), 0.1)
        self.optimizer_REALUNET.step()

    def forward_x8(self):
        pass

    def update_before_iter(self):
        self.optimizer_REALUNET.zero_grad()
        self.optimizer_REALUNET.step()
        self.update_learning_rate()

               
class REALUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(REALUNet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

    def forward(self, x):

        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        conv10 = torch.clip(conv10, -1, 1)
        out = conv10
        return out

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
