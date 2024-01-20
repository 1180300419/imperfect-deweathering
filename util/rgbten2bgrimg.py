import torch
import numpy as np


def cvt_RGBTensor2BGRImg(rgb_tensor):
    # rgb_tensor格式为: B * C * H * W，　取值范围是0-1
    # 返回list, list长度为B, list[index]保存HWC的BGR图像,取值范围是0-255    
    img = torch.clamp(rgb_tensor.detach().cpu() * 255, 0, 255).round()
    bgr_imgs = []
    B = img.shape[0]
    for index in range(B):
        # print(img[index].shape)
        bgr_imgs.append(np.array(img[index]).astype(np.uint8).transpose((1, 2, 0))[..., ::-1])
    return bgr_imgs
    