import time
import os

from PIL import Image

from ssim import SSIM
from ssim.utils import get_gaussian_kernel

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

save_dir = '' # directory that contains the restored images
log_path = 'CWSSIM.txt' 
cw_ssim_list = []
cw_ssim_img = []
for i in range(1518):
    start = time.time()
    im_dir = os.path.join(save_dir, str(i))
    im_gt = Image.open(os.path.join(im_dir, 'gt.jpg'))
    for j in [0, 12, 24, 36]:
        im_restore = Image.open(f'{im_dir}/turb_out/{j}.jpg')
        cw_ssim = SSIM(im_gt).cw_ssim_value(im_restore)
        cw_ssim_img.append(cw_ssim)
        
        with open(log_path, 'a') as file:
            file.write(f'{i}, {j}, {cw_ssim} \n')
    cw = sum(cw_ssim_img) / len(cw_ssim_img)
    cw_ssim_list.append(cw)
    all_time = time.time()-start
    cw_ssim_img = []
    print(f'image {i} finished, CW-SSIM score is {cw}, use time {all_time}')

cw_final = sum(cw_ssim_list) / len(cw_ssim_list)
with open(log_path, 'a') as file:
    file.write(f'final CW-SSIM is {cw_final}')
print(f'final CW-SSIM is {cw_final}')
