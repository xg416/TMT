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
    

def inject_noise(img, noise_sigma=0.0001):
    noise = (noise_sigma**0.5)*torch.randn(img.shape)
    out = img + noise
    return out.clamp(0,1)

def inject_noise_np(img, noise_sigma=0.0001):
    img = img.astype(float) / 255
    noise = (noise_sigma**0.5)*np.random.randn(*img.shape)
    out = img + noise
    out = (np.clip(out,0,1) * 255.0)
    return out.round().astype(np.uint8)
      
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
                att_type='sep').cuda().train()

# load
checkpoint = torch.load('TMT_sep_v.pth')
net.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
for name, param in net.named_parameters():
    param.requires_grad = False
    
# load video
# for kk in range(1,28):
kk=19
# vpath = f'/media/xingguang/Seagate Backup Plus Drive/Turb_data/turbulence_dataset/integrated/real_video/video_{kk}_all.avi'
vpath = '/home/xingguang/Documents/turb/datasets/complex_cnn/data_CLEAR/Heat Haze East Midlands Airport_distorted.mp4'
turb_vid = cv2.VideoCapture(vpath)

h, w = int(turb_vid.get(4)), int(turb_vid.get(3))
fps = int(turb_vid.get(5))
total_frames = int(turb_vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 400
# total_frames = 400
print(total_frames)
test_frame_info = [{'start':0, 'range':[0,9]}]
num_chunk = (total_frames-1) // 6
for i in range(1, num_chunk):
    if i == num_chunk-1:
        test_frame_info.append({'start':total_frames-12, 'range':[i*6+3,total_frames]})
    else:
        test_frame_info.append({'start':i*6, 'range':[i*6+3,i*6+9]})

# all_frames = [turb_vid.read()[1] for i in range(total_frames)]
frame_path = '/home/xingguang/Documents/turb/datasets/complex_cnn/data_CLEAR/Heat Haze East Midlands Airport_distorted'
all_frames = [cv2.imread(os.path.join(frame_path, '{:05d}.png'.format(i+401))) for i in range(total_frames)]
# hh = 270
# ww = 480
# all_frames = [cv2.resize(f, (hh,ww)) for f in all_frames]
hh, ww = h, w
patch_size = 256

inp_frames = []
out_frames = []
turb_vid.release()

start_t = time.time()
for i in range(num_chunk):
    print(f'{i}/{num_chunk}')
    out_range = test_frame_info[i]['range']
    in_range = [test_frame_info[i]['start'], test_frame_info[i]['start']+12]
    inp_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in all_frames[in_range[0]:in_range[1]]]
    inp_imgs = [TF.to_tensor(TF.resize(img, (hh, ww))) for img in inp_imgs]
    # inp_imgs = [inject_noise(img) for img in inp_imgs]
    input_ = torch.stack(inp_imgs, dim=1).unsqueeze(0).cuda()
    if max(hh,ww)<patch_size:
        recovered = net(input_)
        recovered = net(recovered)
        recovered = net(recovered)
    else:
        recovered = test_spatial_overlap(input_, net, patch_size)
    recovered = recovered.permute(0,2,1,3,4)
    input_  = input_.permute(0,2,1,3,4)
    for j in range(out_range[0]-in_range[0], out_range[1]-in_range[0]):
        out = cv2.cvtColor(restore_PIL(recovered, 0, j), cv2.COLOR_RGB2BGR)
        out_frames.append(out)
        
    for j in range(out_range[0]-in_range[0], out_range[1]-in_range[0]):
        inp = cv2.cvtColor(restore_PIL(input_, 0, j), cv2.COLOR_RGB2BGR)
        inp_frames.append(inp)

print(f'total time consumption is {time.time()-start_t} s')

for fid, frame in enumerate(out_frames):
    oframe = np.concatenate((inp_frames[fid], frame), axis=1)
    cv2.imwrite('./TMT_airport/{:04d}.png'.format(fid+401), oframe)

# output_writer = cv2.VideoWriter('TMT_airport_after400.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (ww*2,hh))
# for fid, frame in enumerate(out_frames):
#     output_writer.write(np.concatenate((inp_frames[fid], frame), axis=1))
# output_writer.release() 
