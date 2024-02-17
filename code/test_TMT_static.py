import argparse
from PIL import Image
import logging
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from model.TMT import TMT_MS
import utils.losses as losses
from utils import utils_image as util
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from data.dataset_video_train import DataLoaderTurbImageTest


def restore_PIL(tensor, b, fidx):
    img = tensor[b, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = img * 255.0  # float32 
    return img


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


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--temp_window', type=int, default=12, help='load frames for a single sequence')
    parser.add_argument('--total_frames', type=int, default=48, help='load frames for a single sequence')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=288, help='Patch size')
    parser.add_argument('--data_path', '-data', type=str, default='/home/zhan3275/data/syn_static/test_turb', help='path of training imgs')
    parser.add_argument('--result_path', '-result', type=str, default='/home/zhan3275/data/simulated_data/test_turb_result', help='path of validation imgs')
    parser.add_argument('--model_path', '-mp', type=str, default=False, help='Load model from a .pth file')
    return parser.parse_args()


args = get_args()
input_dir = args.data_path
result_dir = args.result_path
model_path = args.model_path
patch_size = args.patch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)
    
logging.basicConfig(filename=f'{result_dir}/result.log', level=logging.INFO, format='%(levelname)s: %(message)s')

test_dataset = DataLoaderTurbImageTest(rgb_dir=input_dir, num_frames=args.total_frames, total_frames=50)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=False, pin_memory=True)

model = TMT_MS(num_blocks=[2,3,3,4], num_refinement_blocks=2, n_frames=args.temp_window, att_type='shuffle').to(device)
        
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
model.eval()

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
        input_all = data[0].to(device)
        gt = data[1].to(device)
        for i in range(args.total_frames//args.temp_window):
            input_ = input_all[:, i*args.temp_window:(i+1)*args.temp_window, ...]
            recovered = test_spatial_overlap(input_.permute(0,2,1,3,4), model, patch_size).permute(0,2,1,3,4)
                    
            img_result_dir = os.path.join(result_dir, str(seq_idx))
            os.makedirs(img_result_dir, exist_ok=True)
            img_result_recover_dir = os.path.join(img_result_dir, 'turb_out')
            os.makedirs(img_result_recover_dir, exist_ok=True)
            img_result_task_dir = os.path.join(img_result_dir, 'turb')
            os.makedirs(img_result_task_dir, exist_ok=True)
            psnr_img = []
            psnr_y_img = []
            ssim_img = []
            ssim_y_img = []
            lpips_img = []
            
            img_gt_01 = gt[b, ...].data.float().clamp_(0, 1)
            if img_gt_01.ndim == 3:
                img_gt = np.transpose(img_gt_01.cpu().numpy(), (1, 2, 0))  # CHW-RGB to HWC-BGR
            else:
                img_gt = img_gt_01.cpu().numpy()
            img_gt = img_gt * 255.0
            
            for fidx in range(input_.shape[1]):
                out = restore_PIL(recovered, b, fidx)
                inp_img = restore_PIL(input_, b, fidx)

                img_rec = recovered[b, fidx, ...].data.unsqueeze(0).clamp_(0, 1)
                psnr_img.append(util.calculate_psnr(out, img_gt, border=0))
                ssim_img.append(util.calculate_ssim(out, img_gt, border=0))
                #print(img_rec.device, img_gt_01.device)
                lpips_img.append(tmf_lpips(img_rec*2-1, img_gt_01.unsqueeze(0)*2-1).item())
                if img_gt.ndim == 3:  # RGB image
                    psnr_y_img.append(util.calculate_psnr(util.mybgr2ycbcr(out), util.mybgr2ycbcr(img_gt), border=0))
                    ssim_y_img.append(util.calculate_ssim(util.mybgr2ycbcr(out), util.mybgr2ycbcr(img_gt), border=0))
                else:
                    psnr_y_img = psnr_img
                    ssim_y_img = ssim_img
                task_save = Image.fromarray(inp_img.round().astype(np.uint8)).convert('RGB')
                task_save.save(os.path.join(img_result_task_dir, f'{i*args.temp_window + fidx}.jpg'), "JPEG")
                out_save = Image.fromarray(out.round().astype(np.uint8)).convert('RGB')
                out_save.save(os.path.join(img_result_recover_dir, f'{i*args.temp_window + fidx}.jpg'), "JPEG")
            logging.info(f'index:{seq_idx}, psnr:{sum(psnr_img)/len(psnr_img)}, ssim:{sum(ssim_img)/len(ssim_img)}, lpips:{sum(lpips_img)/len(lpips_img)}, \
                psnry:{sum(psnr_y_img)/len(psnr_y_img)}, ssimy:{sum(ssim_y_img)/len(ssim_y_img)}')
            psnr.append(sum(psnr_img)/len(psnr_img))
            ssim.append(sum(ssim_img)/len(ssim_img))
            psnry.append(sum(psnr_y_img)/len(psnr_y_img))
            ssimy.append(sum(ssim_y_img)/len(ssim_y_img))
            lpips.append(sum(lpips_img)/len(lpips_img))
        
        
        gt_save = Image.fromarray(img_gt.round().astype(np.uint8)).convert('RGB')
        gt_save.save(os.path.join(img_result_dir, 'gt.jpg'), "JPEG")
        seq_idx += 1  
    logging.info(f'Overall psnr:{sum(psnr)/len(psnr)}, ssim:{sum(ssim)/len(ssim)}, psnry:{sum(psnry)/len(psnry)}, ssimy:{sum(ssimy)/len(ssimy)}, lpips:{sum(lpips)/len(lpips)}')
