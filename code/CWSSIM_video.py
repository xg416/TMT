import time
import os
import cv2
import numpy as np
from PIL import Image

from ssim import SSIM
from ssim.utils import get_gaussian_kernel

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

save_dir = '' # directory that contains restored videos
gt_dir = '/turb_syn_videos/test/gt/'   # directory that contains GT videos
log_path = 'CWSSIM.txt' 

cw_ssim_list = []
cw_ssim_img = []
for v_name in os.listdir(gt_dir):
    start = time.time()
    gt_path = os.path.join(gt_dir, v_name)
    save_path = os.path.join(save_dir, v_name)
    cap_gt = cv2.VideoCapture(gt_path)
    cap_result = cv2.VideoCapture(save_path)
    
    ret_gt, frame_gt = cap_gt.read()
    ret_r, frame_r = cap_result.read()
    while ret_gt and ret_r:
        im_gt = Image.fromarray((255*frame_gt).astype(np.uint8))
        im_r = Image.fromarray((255*frame_r).astype(np.uint8))
        cw_ssim = SSIM(im_gt).cw_ssim_value(im_r)
        cw_ssim_img.append(cw_ssim)
        ret_gt, frame_gt = cap_gt.read()
        ret_r, frame_r = cap_result.read()    
    cap_gt.release()
    cap_result.release()
    
    cw = sum(cw_ssim_img) / len(cw_ssim_img)
    cw_ssim_list.append(cw)
    all_time = time.time()-start
    cw_ssim_img = []
    with open(log_path, 'a') as file:
        file.write(f'{v_name}: {cw} \n')
    print(f'video {v_name} finished, CW-SSIM score is {cw}, use time {all_time}')

cw_final = sum(cw_ssim_list) / len(cw_ssim_list)
with open(log_path, 'a') as file:
    file.write(f'final CW-SSIM is {cw_final}')
print(f'final CW-SSIM is {cw_final}')
