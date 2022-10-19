import cv2
import os
import os.path as osp
from matplotlib import rc_params

import numpy as np

l_dir = "../checkpoints/dwt/test_epoch_20/inter_rgb_out"
r_dir = "../checkpoints/unet-4resblocks-2e-4_rainaug/test_epoch_20/inter_rgb_out"
# r_dir = "../checkpoints/gt-rain/test_epoch_20/inter_rgb_out_4"
dst_dir = "../checkpoints/rgb_out_merge"

os.makedirs(dst_dir, exist_ok=True)

# for scene in os.listdir(l_dir):
#     for img_name in os.listdir(os.path.join(l_dir, scene)):
#         if img_name in os.listdir(os.path.join(r_dir, scene)) and img_name[-3:] == 'png':
#             l_img = cv2.imread(osp.join(l_dir, scene, img_name))
#             # m_img = cv2.imread(osp.join(middle_dir, img_name))
#             r_img = cv2.imread(osp.join(r_dir, scene, img_name))
#             img = np.hstack([l_img, r_img])
#             os.makedirs(os.path.join(dst_dir, scene), exist_ok=True)
#             cv2.imwrite(osp.join(dst_dir, scene, img_name), img)
#             print(osp.join(dst_dir, scene, img_name))

os.makedirs(dst_dir, exist_ok=True)

for img_name in os.listdir(l_dir):
    # print(img_name)
    if img_name in os.listdir(os.path.join(r_dir)) and img_name[-3:] == 'png':
        l_img = cv2.imread(osp.join(l_dir, img_name))
        # m_img = cv2.imread(osp.join(middle_dir, img_name))
        r_img = cv2.imread(osp.join(r_dir, img_name))
        img = np.hstack([l_img, r_img])
        os.makedirs(os.path.join(dst_dir), exist_ok=True)
        cv2.imwrite(osp.join(dst_dir, img_name), img)
        # print(osp.join(dst_dir, img_name))