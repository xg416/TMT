import cv2
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from models.DeTurbT import RT_warp
import time

def split_to_patches(h, w, s):
    nh = h // s + 1
    nw = w // s + 1
    ol_h = int((nh * s - h) / (nh - 1))
    ol_w = int((nw * s - w) / (nw - 1))
    h_start = 0
    w_start = 0
    hpos = [h_start]
    wpos = [w_start]
    for i in range(1, nh):
        h_start = hpos[-1] + s - ol_h
        if h_start+s > h:
            h_start = h-s
        hpos.append(h_start)
    for i in range(1, nw):
        w_start = wpos[-1] + s - ol_w
        if w_start+s > w:
            w_start = w-s
        wpos.append(w_start)
    return hpos, wpos
    
def test_spatial_overlap(input_blk, model, patch_size):
    _,c,l,h,w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros_like(input_blk)
    out_masks = torch.zeros_like(input_blk)
    for hi in hpos:
        for wi in wpos:
            input_ = input_blk[..., hi:hi+patch_size, wi:wi+patch_size]
            output_ = model(input_)
            out_spaces[..., hi:hi+patch_size, wi:wi+patch_size].add_(output_)
            out_masks[..., hi:hi+patch_size, wi:wi+patch_size].add_(torch.ones_like(input_))
    return out_spaces / out_masks

def restore_PIL(tensor, b, fidx):
    img = tensor[b, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return img   
    
    
torch.cuda.set_device(0)
net = RT_warp(num_blocks=[2,3,3,4], 
                heads=[1,2,4,8], 
                num_refinement_blocks=2, 
                warp_mode='none', 
                n_frames=12, 
                att_type='shuffleori').cuda().eval()

# summary(net, (3,10,64,64))
# summary(net, (3,12,64,64))
# # load
checkpoint = torch.load('shuffle_static.pth')
net.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
for name, param in net.named_parameters():
    param.requires_grad = False


# img_path = '/home/xingguang/Documents/turb/datasets/complex_cnn/data_CLEAR/flash_distorted'
video_path = '/home/xingguang/Documents/turb/datasets/complex_cnn/data_CLEAR/hillhouse_distorted.mp4'
turb_vid = cv2.VideoCapture(video_path)
total_frames = int(turb_vid.get(cv2.CAP_PROP_FRAME_COUNT))

all_frames = [turb_vid.read()[1] for i in range(total_frames)]
turb_vid.release()
turb_imgs = [all_frames[n*6] for n in range(12)]

save_path = f'./hillhouse_restorted/TMT_shuffle'
os.makedirs(save_path, exist_ok=True)


# turb_imgs = [Image.open(p) for p in turb_img_list]
turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
input_ = torch.stack(turb_imgs, dim=1).unsqueeze(0).cuda()

recovered = test_spatial_overlap(input_, net, 240)
recovered = recovered.permute(0,2,1,3,4)
fused = recovered.sum(dim=1, keepdim=True) / 12
outfused = restore_PIL(fused, 0, 0)
out_save = Image.fromarray(outfused).convert('RGB')
out_save.save(os.path.join(f'./hillhouse_restorted/TMT_shuffle.jpg'), "JPEG")
for fidx in range(input_.shape[2]):
    out = restore_PIL(recovered, 0, fidx)
    # out_frames.append(out)
    out_save = Image.fromarray(out).convert('RGB')
    out_save.save(os.path.join(save_path, f'{fidx}.jpg'), "JPEG")