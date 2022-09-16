import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
import numpy as np
import math
from .ISP.isp_model import ISP, ISP4test


# using NAFNet structure
class NAF2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(NAF2Model, self).__init__(opt)

        self.opt = opt
        self.loss_names = ['NAF2_L1_RAW', 'Total']
        # self.loss_names = ['NAF2_L1_RAW', 'NAF2_PSNR_RAW','NAF2_SWD_RAW','NAF2_LPIPS_RGB','NAF2_KL_RAW', 'NAF2_L1_RGB','NAF2_SWD_RGB','NAF2_VGG_RGB', 'Total',
        # 'NAF2_SSIM_RGB']
        # self.loss_names = ['NAF2_PSNR_RGB','NAF2_SSIM_RGB','NAF2_LPIPS_RGB','NAF2_KL_RAW','Total']
        if self.isTrain == True:
            self.visual_names = ['gt_raw_img', 'gt_rgb_img', 'data_out_raw', 'noise_dbinb_img', 'noise_dbinc_img', 'data_out_rgb']
        else:
            if 'real' in opt.split: 
                self.visual_names = ['data_out_raw','data_out_rgb']
            else:
                self.visual_names = ['gt_raw_img', 'gt_rgb_img', 'data_out_raw', 'noise_dbinb_img', 'noise_dbinc_img', 'data_out_rgb']
        self.model_names = ['NAF2'] 
        self.optimizer_names = ['NAF2_optimizer_%s' % opt.optimizer]

        # naf = NAF2(opt, width=64, enc_blk_nums=[2, 2, 4, 8])
        naf = NAF2(opt, width=64, middle_blk_num=24)
        # naf = NAF2(opt, width=64)
        # naf = NAF2(opt, width=128, enc_blk_nums=[2, 4, 10, 24])
        self.netNAF2= N.init_net(naf, opt.init_type, opt.init_gain, opt.gpu_ids)
        
        self.netISP = ISP(opt.gpu_ids, 'GBRG')
        self.netISP4test = ISP4test(opt.gpu_ids, 'GBRG')

        if self.isTrain:		
            self.optimizer_NAF2 = optim.AdamW(self.netNAF2.parameters(),
                                lr=opt.lr,
                                betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)
        
            self.optimizers = [self.optimizer_NAF2]

            self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
            self.criterionMSE = N.init_net(L.MSELoss(), gpu_ids=opt.gpu_ids)
            self.criterionTV = N.init_net(L.TVLoss(), gpu_ids=opt.gpu_ids)
            self.criterionSWD = N.init_net(L.SWDLoss(), gpu_ids=opt.gpu_ids)
            self.criterionSWDRAW = N.init_net(L.SWD(), gpu_ids=opt.gpu_ids)
            self.criterionPSNR = N.init_net(L.PSNRLoss(), gpu_ids=opt.gpu_ids)
            self.criterionSSIM = N.init_net(L.SSIMLoss(), gpu_ids=opt.gpu_ids)
            self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
            self.criterionKL = N.init_net(L.KL_Loss(), gpu_ids=opt.gpu_ids)
            self.criterionLPIPS = N.init_net(L.LPIPSLoss(), gpu_ids=opt.gpu_ids)

    def set_input(self, input, epoch):
        self.epoch = epoch
        if epoch != -2:
            self.gt_raw_img = input['gt_raw_img'].to(self.device)
            self.gt_rgb_img = input['gt_rgb_img'].to(self.device)
            self.gt_img = input['gt_rgb_img'].to(self.device)
            self.gt_rgb_img = torch.clip(torch.pow(self.gt_rgb_img, 2.2),0,1)
        else:
            pass
        self.noise_dbinb_img = input['noise_dbinb_img'].to(self.device)
        self.noise_dbinc_img = input['noise_dbinc_img'].to(self.device)
        self.image_paths = input['fname']
        self.r_gain = input['r_gain'].to(self.device)
        self.b_gain = input['b_gain'].to(self.device)
        self.CCM = input['CCM'].to(self.device)

        if self.isTrain or not self.opt.self_ensemble:
            self.noise_img = pack_raw_image(self.noise_dbinb_img, self.noise_dbinc_img, mode='0')
        # self.noise_img, self.loc_img = pack_raw_image(self.noise_img, mode='0')

    # for test
    def forward(self):
        if self.opt.self_ensemble and not self.isTrain:
            print('x8')
            self.data_out_raw = self.forward_x8(self.noise_dbinb_img, self.noise_dbinc_img, self.netNAF2)
            # self.data_out_raw = self.forward_x8(self.noise_img, self.netRCAN)
            self.data_out_rgb = self.netISP4test(self.data_out_raw, self.r_gain, self.b_gain, self.CCM)
        else:
            if self.epoch > 0: 
                # self.data_out_raw = self.netRCAN(self.noise_img,self.loc_img)
                self.data_out_raw = self.netNAF2(self.noise_img)
                self.data_out_rgb = self.netISP(self.data_out_raw, self.r_gain, self.b_gain, self.CCM)
            else:
                # self.data_out_raw = self.netRCAN(self.noise_img,self.loc_img)
                self.data_out_raw = self.netNAF2(self.noise_img)
                self.data_out_rgb = self.netISP4test(self.data_out_raw, self.r_gain, self.b_gain, self.CCM)

    def backward(self):
        self.loss_NAF2_LPIPS_RGB = 0 #self.criterionLPIPS(self.data_out_rgb, self.gt_rgb_img).mean()
        self.loss_NAF2_L1_RAW = self.criterionL1(self.data_out_raw, self.gt_raw_img).mean()
        self.loss_NAF2_KL_RAW = 0#self.criterionKL(self.data_out_raw, self.gt_raw_img).mean()
        self.loss_NAF2_PSNR_RAW = 0#self.criterionPSNR(self.data_out_raw, self.gt_raw_img).mean()  
        self.loss_NAF2_L1_RGB = 0 # self.criterionL1(self.data_out_rgb, self.gt_rgb_img).mean()
        self.loss_NAF2_SWD_RGB = 0#self.criterionSWD(self.data_out_rgb, self.gt_rgb_img).mean()
        self.loss_NAF2_VGG_RGB = 0#self.criterionVGG(self.data_out_rgb, self.gt_rgb_img).mean()
        self.loss_NAF2_SSIM_RGB = 0#self.criterionSSIM(self.data_out_rgb, self.gt_rgb_img).mean()
        self.loss_NAF2_SSIM_RGB = 0#1 - self.loss_NAF2_SSIM_RGB
       
        # self.loss_Total = self.loss_NAF2_L1_RAW + self.loss_NAF2_L1_RGB + 0.01 * self.loss_NAF2_SWD_RGB +\
        #         0.3*self.loss_NAF2_SSIM_RGB  + self.loss_NAF2_PSNR_RAW +  0.02*self.loss_NAF2_LPIPS_RGB +\
        #         self.loss_NAF2_KL_RAW + self.loss_NAF2_VGG_RGB
        self.loss_Total = self.loss_NAF2_L1_RAW
        self.loss_Total.backward()

    def optimize_parameters(self):
        # watch for nan
        # print('head weight',self.netRCAN.module.head.weight[0,0,:10,:10])
        # print('head bias',self.netRCAN.module.head.bias)
        self.forward()
        # for param in self.netRCAN.module.head.parameters():
        #     print('here')
        #     print(param[0,0,:10,:10])
        self.optimizer_NAF2.zero_grad()
        self.backward()
        torch.nn.utils.clip_grad_norm_(self.netNAF2.parameters(), 0.1)
        self.optimizer_NAF2.step()
    
    def forward_x8(self, noise_dbinb, noise_dbinc, forward_function):
        def _transform(v, op):
            b, c, h, w = v.shape
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[..., ::-1]
                out = np.zeros((b, c, h, w))
                out[..., 0:w-1] = tfnp[..., 1:w]
                out[..., w-1] = out[..., w-3]

            if op == 'h':
                tfnp = v2np[..., ::-1, :]
                out = np.zeros((b, c, h, w))
                out[..., 0:h-1, :] = tfnp[..., 1:h, :]
                out[..., h-1, :] = out[..., h-3, :]
            out = out.copy()
            ret = torch.Tensor(out).to(self.device)
            return ret

        def _transform_inverse(v, op):
            v2np = v.data.cpu().numpy()
            b, c, h, w = v2np.shape
            if op == 'v':
                out = v2np[..., ::-1]
                out[..., 0:w-1] = out[..., 1:w]
                out[..., w - 1] = out[..., w - 3]
            if op == 'h':
                out = v2np[..., ::-1, :]
                out[..., 0:h-1, :] = out[..., 1:h, :]
                out[..., h - 1, :] = out[..., h - 3, :]
            out = out.copy()
            ret = torch.Tensor(out).to(self.device)
            return ret
        h, w = noise_dbinb.shape[-2:]
        dbinb_list = [noise_dbinb]
        dbinc_list = [noise_dbinc]
        
        for tf in 'v', 'h':
            # dbinb_list.append(_transform(t, tf) for t in dbinb_list)
            # dbinc_list.append(_transform(t, tf) for t in dbinc_list)
            dbinb_list.append(_transform(noise_dbinb, tf))
            dbinc_list.append(_transform(noise_dbinc, tf))

        lr_list = []
        for i in range(len(dbinb_list)):
            # print(dbinb_list[i].shape)
            lr_list.append(pack_raw_image(dbinb_list[i], dbinc_list[i], mode='0'))

        sr_list = [forward_function(aug) for aug in lr_list]
        b, c, h, w = sr_list[0].shape
        mask_list = []
        for i in range(len(sr_list)):
            #print(i, sr_list[i].shape)
            mask = torch.zeros((b, c, h, w), dtype=sr_list[0].dtype, device=sr_list[0].device)
            if i == 0:
                sr_list[i] = sr_list[i]
                mask[..., :] = 1
                mask_list.append(mask)
            if i == 1:
                # output = torch.zeros((b, c, h, w), dtype=sr_list[0].dtype,device=sr_list[0].device)
                sr_list[i] = _transform_inverse(sr_list[i], 'v')
                mask[..., 0:w-1] = 1
                mask_list.append(mask)
            if i == 2:
                sr_list[i] = _transform_inverse(sr_list[i], 'h')
                # print('sr_list', sr_list[i].shape)
                mask[..., 0:h-1, :] = 1
                mask_list.append(mask)
            if i == 3:
                sr_list[i] = _transform_inverse(sr_list[i], 'h')
                sr_list[i] = _transform_inverse(sr_list[i], 'v')
                mask[..., 0:h-1, 0:w-1] = 1
                mask_list.append(mask)
           
        output_cat = torch.cat(sr_list, dim=0)
        # mask_cat = torch.cat(mask_list, dim=0)
        # mask = mask_cat.sum(dim=0, keepdim=True)
        # output = output_cat.sum(dim=0, keepdim=True) / mask
        output = output_cat.mean(dim=0, keepdim=True)
        return output

class NAF2(nn.Module):

    def __init__(self, opt, img_channel=2, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        self.opt = opt
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[N.NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[N.NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[N.NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        intro = x
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        # x = x + intro
        x = self.ending(x)

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

operation_seed_counter = 0

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def pack_raw_image(noise_dbinb_img, noise_dbinc_img, bayer='RGBW', mode='0'):  # HxW
    """ Packs a single channel bayer image into 4 channel tensor"""
    # print(noise_dbinb_img.shape)
    im_out = torch.zeros((noise_dbinb_img.shape[0],2, noise_dbinb_img.shape[2], noise_dbinb_img.shape[3]), dtype=noise_dbinb_img.dtype,device=noise_dbinb_img.device)
    # if bayer == 'RGBW':
    #     # R
    #     im_out[..., 0, 2::4, 1::4] = noise_dbinb_img[..., 0, 2::4, 1::4]
    #     im_out[..., 0, 3::4, 0::4] = noise_dbinb_img[..., 0, 3::4, 0::4]
    #     # G
    #     im_out[..., 1, 0::4, 1::4] = noise_dbinb_img[..., 0, 0::4, 1::4]
    #     im_out[..., 1, 1::4, 0::4] = noise_dbinb_img[..., 0, 1::4, 0::4]
    #     im_out[..., 1, 2::4, 3::4] = noise_dbinb_img[..., 0, 2::4, 3::4]
    #     im_out[..., 1, 3::4, 2::4] = noise_dbinb_img[..., 0, 3::4, 2::4]
    #     # B
    #     im_out[..., 2, 0::4, 3::4] = noise_dbinb_img[..., 0, 0::4, 3::4]
    #     im_out[..., 2, 1::4, 2::4] = noise_dbinb_img[..., 0, 1::4, 2::4]
    #     # W
    #     im_out[..., 3, 0::2, 0::2] = noise_dbinb_img[..., 0, 0::2, 0::2]
    #     im_out[..., 3, 1::2, 1::2] = noise_dbinb_img[..., 0, 1::2, 1::2]
    # im_loc = torch.zeros((noise_dbinb_img.shape[0],4, noise_dbinb_img.shape[2], noise_dbinb_img.shape[3]), dtype=noise_dbinb_img.dtype,device=noise_dbinb_img.device)
    # im_loc[im_out!=0] = 1
    # if mode == '0':
    #     pass
    # elif mode == 'rand':
    #     rd = torch.rand(im_out.shape, generator=get_generator(), device=noise_dbinb_img.device )
    #     im_out[im_out==0] = rd[im_out==0]
    # elif mode == 'near':
    #     # fill red
    #     check_br = near_fill_red(im_out[:,0,:,:].clone().unsqueeze(dim=1), im_out[:,0,:,:])
    #     # fill green
    #     check_g = near_fill_green(im_out[:,1,:,:].clone().unsqueeze(dim=1), im_out[:,1,:,:])
    #     # fill blue
    #     check_b = near_fill_blue(im_out[:,2,:,:].clone().unsqueeze(dim=1), im_out[:,2,:,:])
    #     # fill white 
    #     check_w = near_fill_white(im_out[:,3,:,:].clone().unsqueeze(dim=1), im_out[:,3,:,:])
    # else:
        # raise ValueError
    im_out[:, 0, :, :] = noise_dbinb_img[..., 0, :, :]
    im_out[:, 1, :, :] = noise_dbinc_img[..., 0, :, :]
    return im_out  # 4xHxW

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

def near_fill_white(white,new_white, mode='rand'):
    if mode == 'rand':
        n, c, h, w = white.shape
        assert c == 1
        mask = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                            dtype=white.dtype,
                            device=white.device)
        idx_pair = torch.tensor([[1,2]], dtype=torch.int64,  device=white.device)
        interpolate_pair = torch.tensor([[0,3], [3, 0]], dtype=torch.int64, device=white.device)    
        rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                            dtype=torch.int64,
                            device=white.device)
        torch.randint(low=0,
                  high=2,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
        mask_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                            dtype=torch.int64,
                            device=white.device)
        mask_pair_idx = idx_pair[mask_idx]
        mask_pair_idx += torch.arange(start=0,
                            end=n * h // 2 * w // 2 * 4,
                            step=4,
                            dtype=torch.int64,
                            device=white.device).reshape(-1, 1)
        interplate_pair_idx = interpolate_pair[rd_idx]
        interplate_pair_idx += torch.arange(start=0,
                            end=n * h // 2 * w // 2 * 4,
                            step=4,
                            dtype=torch.int64,
                            device=white.device).reshape(-1, 1)
        mask = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                    dtype=torch.float32,
                    device=white.device)
        # get masks
        mask[mask_pair_idx] = 1
        mask =  F.pixel_shuffle(mask.reshape(
            n, h //2, w // 2, 4).permute(0, 3, 1, 2),2)
        img_per_channel = space_to_depth(white, block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        img_per_channel[mask_pair_idx] =  img_per_channel[interplate_pair_idx]
        res = F.pixel_shuffle(img_per_channel.reshape(
            n, h //2, w // 2, 4).permute(0, 3, 1, 2),2) 
        assert res.shape[1] == 1
        new_white = res.unsqueeze(dim=1)  
        return mask           
    else:
        raise ValueError

def near_fill_green(g, new_g, mode='fix'):
    if mode == 'fix':
        n, c, h, w = g.shape
        assert c == 1
        mask = torch.zeros(size=(n * h // 4 * w // 4 * 16, ),
                            dtype=g.dtype,
                            device=g.device)
        idx_pair = torch.tensor([[0,2,3,5,6,7,8,9,10,12,13,15]], dtype=torch.int64,  device=g.device)
        interpolate_pair = torch.tensor([[1,1,1,4,1,11,4,4,11,4,14,14]], dtype=torch.int64, device=g.device)    
        mask_idx = torch.zeros(size=(n * h // 4 * w // 4, ),
                            dtype=torch.int64,
                            device=g.device)
        mask_pair_idx = idx_pair[mask_idx]
        mask_pair_idx += torch.arange(start=0,
                    end=n * h // 4 * w // 4 * 16,
                    step=16,
                    dtype=torch.int64,
                    device=g.device).reshape(-1, 1)
        interplate_pair_idx = interpolate_pair[mask_idx]
        interplate_pair_idx += torch.arange(start=0,
                            end=n * h // 4 * w // 4 * 16,
                            step=16,
                            dtype=torch.int64,
                            device=g.device).reshape(-1, 1)
        mask = torch.zeros(size=(n * h // 4 * w // 4 * 16, ),
                    dtype=torch.float32,
                    device=g.device)
        # get masks
        mask[mask_pair_idx] = 1
        mask =  F.pixel_shuffle(mask.reshape(
            n, h //4, w // 4, 16).permute(0, 3, 1, 2),4) 
        img_per_channel = space_to_depth(g, block_size=4)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        img_per_channel[mask_pair_idx] =  img_per_channel[interplate_pair_idx]
        res = F.pixel_shuffle(img_per_channel.reshape(
            n, h //4, w // 4, 16).permute(0, 3, 1, 2),4)
        assert res.shape[1] == 1
        new_g = res.unsqueeze(dim=1)           
        return mask            
    else:
        raise ValueError

def near_fill_blue(b,new_b,mode='fix'):
    if mode == 'fix':
        n, c, h, w = b.shape
        assert c == 1
        mask = torch.zeros(size=(n * h // 4 * w // 4 * 16, ),
                            dtype=b.dtype,
                            device=b.device)
        idx_pair = torch.tensor([[0,1,2,4,5,7,8,9,10,11,12,13,14,15]], dtype=torch.int64,  device=b.device)
        interpolate_pair = torch.tensor([[3,3,3,6,6,3,6,6,6,6,6,6,6,6]], dtype=torch.int64, device=b.device)    
        mask_idx = torch.zeros(size=(n * h // 4 * w // 4, ),
                            dtype=torch.int64,
                            device=b.device)
        mask_pair_idx = idx_pair[mask_idx]
        mask_pair_idx += torch.arange(start=0,
                    end=n * h // 4 * w // 4 * 16,
                    step=16,
                    dtype=torch.int64,
                    device=b.device).reshape(-1, 1)
        interplate_pair_idx = interpolate_pair[mask_idx]
        interplate_pair_idx += torch.arange(start=0,
                            end=n * h // 4 * w // 4 * 16,
                            step=16,
                            dtype=torch.int64,
                            device=b.device).reshape(-1, 1)
        mask = torch.zeros(size=(n * h // 4 * w // 4 * 16, ),
                    dtype=torch.float32,
                    device=b.device)
        # get masks
        mask[mask_pair_idx] = 1
        mask =  F.pixel_shuffle(mask.reshape(
            n, h //4, w // 4, 16).permute(0, 3, 1, 2),4)
        # print(new_b.shape[1])
        img_per_channel = space_to_depth(b, block_size=4)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        img_per_channel[mask_pair_idx] =  img_per_channel[interplate_pair_idx]
        res = F.pixel_shuffle(img_per_channel.reshape(
            n, h //4, w // 4, 16).permute(0, 3, 1, 2),4)
        assert res.shape[1] == 1
        new_b = res.unsqueeze(dim=1)  
        return mask            
    else:
        raise ValueError

def near_fill_red(r, new_r, mode='fix'):
    if mode == 'fix':
        n, c, h, w = r.shape
        assert c == 1
        mask = torch.zeros(size=(n * h // 4 * w // 4 * 16, ),
                            dtype=r.dtype,
                            device=r.device)
        idx_pair = torch.tensor([[0,1,2,3,4,5,6,7,8,10,11,13,14,15]], dtype=torch.int64,  device=r.device)
        interpolate_pair = torch.tensor([[9,9,9,9,9,9,9,9,12,9,9,12,12,12]], dtype=torch.int64, device=r.device)  
        mask_idx = torch.zeros(size=(n * h // 4 * w // 4, ),
                            dtype=torch.int64,
                            device=r.device)
        mask_pair_idx = idx_pair[mask_idx]
        mask_pair_idx += torch.arange(start=0,
                    end=n * h // 4 * w // 4 * 16,
                    step=16,
                    dtype=torch.int64,
                    device=r.device).reshape(-1, 1)
        interplate_pair_idx = interpolate_pair[mask_idx]
        interplate_pair_idx += torch.arange(start=0,
                            end=n * h // 4 * w // 4 * 16,
                            step=16,
                            dtype=torch.int64,
                            device=r.device).reshape(-1, 1)
        mask = torch.zeros(size=(n * h // 4 * w // 4 * 16, ),
            dtype=torch.float32,
            device=r.device)
        # get masks
        mask[mask_pair_idx] = 1
        mask =  F.pixel_shuffle(mask.reshape(
            n, h //4, w // 4, 16).permute(0, 3, 1, 2),4)
        img_per_channel = space_to_depth(r, block_size=4)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        img_per_channel[mask_pair_idx] =  img_per_channel[interplate_pair_idx]
        res = F.pixel_shuffle(img_per_channel.reshape(
            n, h //4, w // 4, 16).permute(0, 3, 1, 2),4)
        assert res.shape[1] == 1 
        new_r =  res.unsqueeze(dim=1)  
        return mask            
    else:
        raise ValueError