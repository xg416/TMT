import argparse
import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from model.TMT_MS import TMT_MS
from model.UNet3d_TMT import DetiltUNet3DS
import cv2
from utils import utils_image as util
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from data.dataset_video_train import DataLoaderTurbVideoTest

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--iteration', '-iter', type=int, default=1, help='testing loop')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=288, help='Patch size')
    parser.add_argument('--temp_window', type=int, default=12, help='load frames for a single sequence')
    parser.add_argument('--data_path', '-data', type=str, default='/home/zhan3275/data/syn_video/test', help='path of training imgs')
    parser.add_argument('--result_path', '-result', type=str, default='/home/zhan3275/data/simulated_data/test_TMT_video', help='path of validation imgs')
    parser.add_argument('--path_tilt', '-pt', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--march', type=str, default='normal', help='model architecture')
    parser.add_argument('--model_path', '-mp', type=str, default=False, help='Load model from a .pth file')
    return parser.parse_args()


def restore_PIL(tensor, b, fidx):
    img_tensor = tensor[b, fidx, ...].data.unsqueeze(0).clamp_(0, 1)
    img = img_tensor.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = img * 255.0  # float32 
    return img, img_tensor

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
    if len(hpos)==2 and hpos[0] == hpos[1]:
        hpos = [hpos[0]]
    if len(wpos)==2 and wpos[0] == wpos[1]:
        wpos = [wpos[0]]   
    return hpos, wpos
    
def test_spatial_overlap(input_blk, model, model_tilt, patch_size):
    _,c,l,h,w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros_like(input_blk)
    out_masks = torch.zeros_like(input_blk)
    for hi in hpos:
        for wi in wpos:
            input_ = input_blk[..., hi:hi+patch_size, wi:wi+patch_size]
            _, _, rectified = model_tilt(input_.permute(0,2,1,3,4))
            output_ = model(rectified.permute(0,2,1,3,4))
            out_spaces[..., hi:hi+patch_size, wi:wi+patch_size].add_(output_)
            out_masks[..., hi:hi+patch_size, wi:wi+patch_size].add_(torch.ones_like(input_))
    return out_spaces / out_masks
    

def temp_segment(total_frames, trunk_len, valid_len):
    residual = (trunk_len - valid_len) // 2
    test_frame_info = [{'start':0, 'range':[0,trunk_len-residual]}]
    num_chunk = (total_frames-1) // valid_len
    for i in range(1, num_chunk+1):
        if i == num_chunk:
            test_frame_info.append({'start':total_frames-args.temp_window, 'range':[i*valid_len+residual,total_frames]})
        elif i*valid_len+trunk_len >= total_frames:
            test_frame_info.append({'start':total_frames-trunk_len, 'range':[i*valid_len+residual, total_frames]})
            break
        else:
            test_frame_info.append({'start':i*valid_len, 'range':[i*valid_len+residual,i*valid_len+trunk_len-residual]})
    return num_chunk, test_frame_info
    
args = get_args()
input_dir = args.data_path
result_dir = args.result_path
model_path = args.model_path
patch_size = args.patch_size
iteration = args.iteration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)
    
#logging.basicConfig(filename=f'{result_dir}/result.log', level=logging.INFO, format='%(levelname)s: %(message)s')
log_file = open(f'{result_dir}/result.log', 'a')
test_dataset = DataLoaderTurbVideoTest(root_dir=input_dir, result_dir=result_dir, max_frame=120, patch_unit=16)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

model = TMT_MS(num_blocks=[2,3,3,4], num_refinement_blocks=2, n_frames=args.temp_window, att_type='shuffle').to(device)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
model.eval()

# load tilt removal model
model_tilt = DetiltUNet3DS(norm='LN', residual='pool', conv_type='dw').to(device)
ckpt_tilt = torch.load(args.path_tilt)
model_tilt.load_state_dict(ckpt_tilt['state_dict'] if 'state_dict' in ckpt_tilt.keys() else ckpt_tilt)
model_tilt.eval()
for param in model_tilt.parameters():
    param.requires_grad = False
    
psnr = []
ssim = []
psnry = []
ssimy = []
lpips = []
tmf_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
b = 0
with torch.no_grad():
    seq_idx = 0
    for data in test_loader:
        out_frames = []
        psnr_video = []
        psnr_y_video = []
        ssim_video = []
        ssim_y_video = []
        lpips_video = []
        input_all = data[0]
        gt = data[1]
        h, w = data[2][b].item(), data[3][b].item()
        total_frames = data[4][b].item()
        fps = data[5][b].item()
        path = os.path.split(data[-1][b])[-1]
        img_result_path = os.path.join(result_dir, path.split('.')[0]+'.mp4')
        if os.path.exists(img_result_path):
            time.sleep(1)
            continue
        num_chunk, test_frame_info = temp_segment(total_frames, args.temp_window, args.temp_window-2)
        for i in range(num_chunk):
            out_range = test_frame_info[i]['range']
            in_range = [test_frame_info[i]['start'], test_frame_info[i]['start']+args.temp_window]
            input_ = input_all[:, in_range[0]:in_range[1], ...]
            input_ = input_.detach().to(device).permute(0,2,1,3,4)
            print(input_.shape, h, w, total_frames, out_range, in_range, input_all.shape, path)

            if max(h, w) >= 272:
                if min(input_.shape[3], input_.shape[4]) < 272:
                    recovered = test_spatial_overlap(input_, model, model_tilt, min(input_.shape[3],input_.shape[4]))
                else:
                    recovered = test_spatial_overlap(input_, model, model_tilt, 272)
            else:
                _, _, rectified = model_tilt(input_.permute(0,2,1,3,4))
                recovered = model(rectified.permute(0,2,1,3,4))

            recovered = recovered[..., :h, :w].permute(0,2,1,3,4)
            gt_local = gt[:, in_range[0]:in_range[1], :, :h, :w]
            for j in range(out_range[0]-in_range[0], out_range[1]-in_range[0]):
                out, out_tensor = restore_PIL(recovered, b, j)
                img_gt, gt_tensor = restore_PIL(gt_local, b, j)
                psnr_video.append(util.calculate_psnr(out, img_gt, border=0))
                ssim_video.append(util.calculate_ssim(out, img_gt, border=0))
                lpips_video.append(tmf_lpips(out_tensor.cuda()*2-1, gt_tensor.cuda()*2-1).item())
                
                if img_gt.ndim == 3:  # RGB image
                    psnr_y_video.append(util.calculate_psnr(util.mybgr2ycbcr(out), util.mybgr2ycbcr(img_gt), border=0))
                    ssim_y_video.append(util.calculate_ssim(util.mybgr2ycbcr(out), util.mybgr2ycbcr(img_gt), border=0))
                else:
                    psnr_y_video = psnr_video
                    ssim_y_video = ssim_video
                out = cv2.cvtColor(out.round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                out_frames.append(out)
        # logging.info(f'video:{path}, psnr:{sum(psnr_video)/len(psnr_video)}, ssim:{sum(ssim_video)/len(ssim_video)}, \
        #    psnry:{sum(psnr_y_video)/len(psnr_y_video)}, ssimy:{sum(ssim_y_video)/len(ssim_y_video)}, lpips:{sum(lpips_video)/len(lpips_video)}')
        with open(f'{result_dir}/result.log', 'a') as log_file:
            log_file.write(f'video:{path}, psnr:{sum(psnr_video)/len(psnr_video)}, ssim:{sum(ssim_video)/len(ssim_video)}, \
                psnry:{sum(psnr_y_video)/len(psnr_y_video)}, ssimy:{sum(ssim_y_video)/len(ssim_y_video)}, lpips:{sum(lpips_video)/len(lpips_video)}\n')
            # log_file.write(f'video:{path}, psnr:{sum(psnr_video)/len(psnr_video)}, ssim:{sum(ssim_video)/len(ssim_video)}, lpips:{sum(lpips_video)/len(lpips_video)}\n')

        psnr.append(sum(psnr_video)/len(psnr_video))
        ssim.append(sum(ssim_video)/len(ssim_video))
        # psnry.append(sum(psnr_y_video)/len(psnr_y_video))
        # ssimy.append(sum(ssim_y_video)/len(ssim_y_video))
        lpips.append(sum(lpips_video)/len(lpips_video))
        output_writer = cv2.VideoWriter(img_result_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w,h))
        for fid, frame in enumerate(out_frames):
            output_writer.write(frame)
        output_writer.release()
        seq_idx + 1
    #logging.info(f'Overall psnr:{sum(psnr)/len(psnr)}, ssim:{sum(ssim)/len(ssim)}, psnry:{sum(psnry)/len(psnry)}, ssimy:{sum(ssimy)/len(ssimy)}')
    with open(f'{result_dir}/result.log', 'a') as log_file:
        log_file.write(f'Overall psnr:{sum(psnr)/len(psnr)}, ssim:{sum(ssim)/len(ssim)}, lpips:{sum(lpips)/len(lpips)}')

