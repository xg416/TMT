import argparse
from PIL import Image
import time
import logging
import os
import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.scheduler import GradualWarmupScheduler
from model.UNet3d_TMT import DetiltUNet3DS
from model.TMT_MS import TMT_MS
import utils.losses as losses
from utils import utils_image as util
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim
from torchmetrics.functional import peak_signal_noise_ratio as tmf_psnr
from utils.general import create_log_folder, get_cuda_info, find_latest_checkpoint
from data.dataset_video_train import DataLoaderTurbVideo

def eval_tensor_imgs(gt, output, input, save_path=None, kw='train', iter_count=0):
    '''
    Input images are 5-D in Batch, length, channel, H, W
    output is list of psnr and ssim
    '''
    psnr_list = []
    ssim_list = []
    for b in range(output.shape[0]):
        for i in range(output.shape[1]):
            img = output[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            img_gt = gt[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            psnr_list.append(tmf_psnr(img, img_gt, data_range=1.0).item())
            ssim_list.append(tmf_ssim(img, img_gt, data_range=1.0).item())
            
            if save_path:
                inp = input[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if inp.ndim == 3:
                    inp = np.transpose(inp, (1, 2, 0))  # CHW-RGB to HWC-BGR
                inp = (inp * 255.0).round().astype(np.uint8)  # float32 to uint8
                
                img = output[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img.ndim == 3:
                    img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
                img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8

                img_gt = gt[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt, (1, 2, 0))  # CHW-RGB to HWC-BGR
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8

                pg_save = Image.fromarray(np.uint8(np.concatenate((inp, img, img_gt), axis=1))).convert('RGB')
                pg_save.save(os.path.join(save_path, f'{kw}_{iter_count}_{b}_{i}.jpg'), "JPEG")
    return psnr_list, ssim_list

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--iters', type=int, default=400000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=240, help='Batch size')
    parser.add_argument('--print-period', '-pp', dest='print_period', type=int, default=1000, help='number of iterations to save checkpoint')
    parser.add_argument('--val-period', '-vp', dest='val_period', type=int, default=5000, help='number of iterations for validation')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('--num_frames', type=int, default=12, help='number of frames for the model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers in dataloader')
    parser.add_argument('--train_path', type=str, default='/home/zhan3275/data/syn_video/train', help='path of training imgs')
    parser.add_argument('--val_path', type=str, default='/home/zhan3275/data/syn_video/test', help='path of validation imgs')
    parser.add_argument('--march', type=str, default='normal', help='model architecture')
    parser.add_argument('--path_tilt', type=str, default=False, help='Load tilt removal model from a .pth file')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--log_path', type=str, default='/home/zhan3275/data/train_log', help='path to save logging files and images')
    parser.add_argument('--task', type=str, default='turb', help='choose turb or blur or both')
    parser.add_argument('--run_name', type=str, default='TMT-dynamic-2stage', help='name of this running')
    parser.add_argument('--start_over', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_path = os.path.join(args.log_path, run_name)
    if not os.path.exists(run_path):
        result_img_path, path_ckpt, path_scipts = create_log_folder(run_path)
    logging.basicConfig(filename=f'{run_path}/recording.log', \
                        level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    get_cuda_info(logging)
    n_frames = args.num_frames
    
    train_dataset = DataLoaderTurbVideo(args.train_path, num_frames=n_frames, patch_size=args.patch_size, noise=0.0001, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    val_dataset = DataLoaderTurbVideo(args.val_path, num_frames=n_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    model = TMT_MS(num_blocks=[2,3,3,4], num_refinement_blocks=2, n_frames=n_frames, att_type='shuffle').to(device)
         
    # load tilt removal model
    model_tilt = DetiltUNet3DS(norm='LN', residual='pool', conv_type='dw').to(device)
    ckpt_tilt = torch.load(args.path_tilt)
    model_tilt.load_state_dict(ckpt_tilt['state_dict'] if 'state_dict' in ckpt_tilt.keys() else ckpt_tilt)
    model_tilt.eval()
    for param in model_tilt.parameters():
        param.requires_grad = False
                
    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.99), eps=1e-8)
    ######### Scheduler ###########
    total_iters = args.iters
    start_iter = 1
    warmup_iter = 5000
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, total_iters-warmup_iter, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_iter, after_scheduler=scheduler_cosine)
    scheduler.step()
    
    
    ######### Resume ###########
    if args.load:
        if args.load == 'latest':
            load_path = find_latest_checkpoint(args.log_path, args.run_name)
            if not load_path:
                print(f'search for the latest checkpoint of {args.run_name} failed!')
        else:
            load_path = args.load
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
        if not args.start_over:
            if 'epoch' in checkpoint.keys():
                start_iter = checkpoint["epoch"] * len(train_dataset)
            elif 'iter' in checkpoint.keys():
                start_iter = checkpoint["iter"] 
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_lr = optimizer.param_groups[0]['lr']
            print('------------------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            logging.info(f'==> Resuming Training with learning rate: {new_lr}')
            print('------------------------------------------------------------------------------')
            
        for i in range(1, start_iter):
            scheduler.step()

    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_count)]).cuda()

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss3D()
    
    logging.info(f'''Starting training:
        Total_iters:     {total_iters}
        Start_iters:     {start_iter}
        Batch size:      {args.batch_size}
        Learning rate:   {new_lr}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {path_ckpt}
    ''')
    
    ######### train ###########
    best_psnr = 0
    best_epoch = 0
    iter_count = start_iter
    model.train()
    for epoch in range(1000000):
        for data in train_loader:
            if iter_count == start_iter:
                current_start_time = time.time()
                current_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder['psnr'] = []
                train_results_folder['ssim'] = []

            # zero_grad
            for param in model.parameters():
                param.grad = None
            if args.task == 'blur':
                input_ = data[0].cuda()
            elif args.task == 'turb':
                input_ = data[1].cuda()
            target = data[2].cuda()
            _, _, rectified = model_tilt(input_)
            output = model(rectified.permute(0,2,1,3,4)).permute(0,2,1,3,4)
            
            if total_iters >= 300000:
                loss = criterion_char(output, target) + 0.05*criterion_edge(output, target)
            else:
                loss = criterion_char(output, target)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            current_loss += loss.item()
            iter_count += 1

            if iter_count % 500 == 0:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_, save_path=result_img_path, kw='train', iter_count=iter_count)
            else:
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
            train_results_folder['psnr'] += psnr_batch
            train_results_folder['ssim'] += ssim_batch
                    
            if iter_count>start_iter and iter_count % args.print_period == 0:
                psnr = sum(train_results_folder['psnr']) / len(train_results_folder['psnr'])
                ssim = sum(train_results_folder['ssim']) / len(train_results_folder['ssim'])
                logging.info('Training: iters {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -Loss {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}'.format(
                    iter_count, total_iters, time.time()-current_start_time, optimizer.param_groups[0]['lr'], current_loss/args.print_period, psnr, ssim))

                torch.save({'iter': iter_count, 
                            'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(path_ckpt, f"model_{iter_count}.pth")) 

                torch.save({'iter': iter_count, 
                            'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(path_ckpt, "latest.pth")) 
                current_start_time = time.time()
                current_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder['psnr'] = []
                train_results_folder['ssim'] = []
                                          
            #### Evaluation ####
            if iter_count>0 and iter_count % args.val_period == 0:
                test_results_folder = OrderedDict()
                test_results_folder['psnr'] = []
                test_results_folder['ssim'] = []

                eval_loss = 0
                model.eval()
                for s, data in enumerate(val_loader):
                    input_ = data[1].to(device)
                    target = data[2].to(device)
                    with torch.no_grad():
                        _, _, rectified = model_tilt(input_)
                        output = model(rectified.permute(0,2,1,3,4)).permute(0,2,1,3,4)
                        loss = criterion_char(output, target)

                        eval_loss += loss.item()
                    
                    if s % 250 == 0:
                        psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_, save_path=result_img_path, kw='val', iter_count=iter_count)
                    else:
                        psnr_batch, ssim_batch = eval_tensor_imgs(target, output, input_)
                    test_results_folder['psnr'] += psnr_batch
                    test_results_folder['ssim'] += ssim_batch
                psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                logging.info('Validation: Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(iter_count, total_iters, eval_loss/s, psnr, ssim))

                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({'iter': iter_count,
                                'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(path_ckpt, "model_best.pth"))
                model.train()
                
if __name__ == '__main__':
    main()
