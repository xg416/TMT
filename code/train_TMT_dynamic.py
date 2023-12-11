import argparse
from PIL import Image
import time
import logging
import os, glob
import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.scheduler import GradualWarmupScheduler, CosineDecayWithWarmUpScheduler
from model.TMT_MS import TMT_MS
import utils.losses as losses
from utils import utils_image as util
from utils.general import create_log_folder, get_cuda_info, find_latest_checkpoint
from data.dataset_video_train import DataLoaderTurbVideo

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--iters', type=int, default=400000, help='Number of iterations for each period')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=240, help='Batch size')
    parser.add_argument('--print-period', '-pp', dest='print_period', type=int, default=1000, help='number of iterations to save checkpoint')
    parser.add_argument('--val-period', '-vp', dest='val_period', type=int, default=5000, help='number of iterations for validation')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('--num_frames', type=int, default=12, help='number of frames for the model')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--train_path', type=str, default='/home/zhan3275/data/syn_video/train', help='path of training imgs')
    parser.add_argument('--val_path', type=str, default='/home/zhan3275/data/syn_video/test', help='path of validation imgs')
    parser.add_argument('--march', type=str, default='normal', help='model architecture')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--log_path', type=str, default='/home/zhan3275/data/train_log', help='path to save logging files and images')
    parser.add_argument('--task', type=str, default='turb', help='choose turb or blur or both')
    parser.add_argument('--run_name', type=str, default='TMT-dynamic', help='name of this running')
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
    
    train_dataset = DataLoaderTurbVideo(args.train_path, num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    val_dataset = DataLoaderTurbVideo(args.val_path, num_frames=args.num_frames, patch_size=args.patch_size, noise=0.0001, is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    model = TMT_MS(num_blocks=[2,3,3,4], num_refinement_blocks=2, n_frames=args.num_frames, att_type='shuffle').to(device)
   
    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.99), eps=1e-8)
    ######### Scheduler ###########
    total_iters = args.iters
    start_iter = 1
    warmup_iter = 10000
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
    # criterion_edge = losses.EdgeLoss3D()
    
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
    s0 = 0
    for epoch in range(1000000):
        for data in train_loader:
            # s1 = time.time()
            if iter_count == start_iter:
                current_start_time = time.time()
                current_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder['psnr'] = []
                train_results_folder['ssim'] = []
                train_results_folder['psnr_y'] = []
                train_results_folder['ssim_y'] = []
            # zero_grad
            for param in model.parameters():
                param.grad = None
            if args.task == 'blur':
                input_ = data[0].cuda()
            elif args.task == 'turb':
                input_ = data[1].cuda()
            target = data[2].cuda()
            output = model(input_.permute(0,2,1,3,4)).permute(0,2,1,3,4)

            loss = criterion_char(output, target)
            # s2 = time.time()
            # loss = criterion_char(output, target) + 0.05*criterion_edge(output, target)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            # s3 = time.time()
            current_loss += loss.item()
            iter_count += 1

            for b in range(output.shape[0]):
                for i in range(output.shape[1]):
                    inp = input_[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if inp.ndim == 3:
                        inp = np.transpose(inp, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    inp = (inp * 255.0).round().astype(np.uint8)  # float32 to uint8
                    
                    img = output[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if img.ndim == 3:
                        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                    # img = np.squeeze(img)
                    
                    img_gt = target[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                    # img_gt = np.squeeze(img_gt)
                
                    train_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                    train_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                    
                    if iter_count % 500 == 0:
                        pg_save = Image.fromarray(np.uint8(np.concatenate((inp, img, img_gt), axis=1))).convert('RGB')
                        pg_save.save(os.path.join(result_img_path, f'train_{iter_count}_{b}_{i}.jpg'), "JPEG")
                                      
                    # if img_gt.ndim == 3:  # RGB image
                    #     img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                    #     img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                    #     train_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                    #     train_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                    # else:
                    #     train_results_folder['psnr_y'] = train_results_folder['psnr']
                    #     train_results_folder['ssim_y'] = train_results_folder['ssim']    
            # s4 = time.time()
            # logging.info(f'img: {s4-s3}, backward: {s3-s2}, forward:{s2-s1}, data: {s1-s0}')
            # s0 = time.time()
            if iter_count>start_iter and iter_count % args.print_period == 0:
                psnr = sum(train_results_folder['psnr']) / len(train_results_folder['psnr'])
                ssim = sum(train_results_folder['ssim']) / len(train_results_folder['ssim'])
                # psnr_y = sum(train_results_folder['psnr_y']) / len(train_results_folder['psnr_y'])
                # ssim_y = sum(train_results_folder['ssim_y']) / len(train_results_folder['ssim_y'])
                # logging.info('Training: iters {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -Loss {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(iter_count, total_iters, time.time()-current_start_time, optimizer.param_groups[0]['lr'], current_loss/args.print_period, psnr, ssim, psnr_y, ssim_y))
                logging.info('Training: iters {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -Loss {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}'.format(iter_count, total_iters, time.time()-current_start_time, optimizer.param_groups[0]['lr'], current_loss/args.print_period, psnr, ssim))
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
                train_results_folder['psnr_y'] = []
                train_results_folder['ssim_y'] = []
                                          
            #### Evaluation ####
            if iter_count>0 and iter_count % args.val_period == 0:
                test_results_folder = OrderedDict()
                test_results_folder['psnr'] = []
                test_results_folder['ssim'] = []
                test_results_folder['psnr_y'] = []
                test_results_folder['ssim_y'] = []
                eval_loss = 0
                model.eval()
                for s, data in enumerate(val_loader):
                    if args.task == 'blur':
                        input_ = data[0].cuda()
                    elif args.task == 'turb':
                        input_ = data[1].cuda()
                    target = data[2].to(device)
                    with torch.no_grad():
                        output = model(input_.permute(0,2,1,3,4)).permute(0,2,1,3,4)
                        loss = criterion_char(output, target)

                        eval_loss += loss.item()
                    
                    for b in range(output.shape[0]):
                        for i in range(output.shape[1]):
                            inp = input_[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                            if inp.ndim == 3:
                                inp = np.transpose(inp, (1, 2, 0))  # CHW-RGB to HWC-BGR
                            inp = (inp * 255.0).round().astype(np.uint8)  # float32 to uint8
                            
                            img = output[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                            if img.ndim == 3:
                                img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
                            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                            # img = np.squeeze(img)
                            
                            img_gt = target[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                            if img_gt.ndim == 3:
                                img_gt = np.transpose(img_gt, (1, 2, 0))  # CHW-RGB to HWC-BGR
                            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                            # img_gt = np.squeeze(img_gt)
                        
                            test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                            test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                                
                            if s % 250 == 0:
                                pg_save = Image.fromarray(np.uint8(np.concatenate((inp, img, img_gt), axis=1))).convert('RGB')
                                pg_save.save(os.path.join(result_img_path, f'val_{iter_count}_{b}_{i}.jpg'), "JPEG")
                                                                     
                            # if img_gt.ndim == 3:  # RGB image
                            #     img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                            #     img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                            #     test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                            #     test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                            # else:
                            #     test_results_folder['psnr_y'] = test_results_folder['psnr']
                            #     test_results_folder['ssim_y'] = test_results_folder['ssim']    
                                
                psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                # psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
                # ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
                
                # logging.info('Validation: Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(iter_count, total_iters, eval_loss/s, psnr, ssim, psnr_y, ssim_y))
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
