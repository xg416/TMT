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
import utils.losses as losses
from utils import utils_image as util
from utils.general import create_log_folder, get_cuda_info
from data.dataset_video_train import DataLoaderTurbVideo

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=6, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=240, help='Batch size')
    parser.add_argument('--num_frames', '-nf', type=int, default=12, help='number of input frames')
    parser.add_argument('--val-period', '-vp', dest='val_period', type=int, default=5, help='number of frames used for reconstruction')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('--train_path', type=str, default='/home/zhan3275/data/data_TMT/turb_syn_videos/train', help='path of training imgs')
    parser.add_argument('--val_path', type=str, default='/home/zhan3275/data/data_TMT/turb_syn_videos/test', help='path of validation imgs')
    parser.add_argument('--load', '-f', type=str, default=False, help='continue with a .pth checkpoint')
    parser.add_argument('--log_path', type=str, default='/home/zhan3275/data/TMT_results/TMT_tilt/', help='path to save logging files')
    parser.add_argument('--run_name', type=str, default='dynamic-tilt', help='name of this experiment')
    parser.add_argument('--start_over', action='store_true', help='start the scheduler from the first epoch')
    return parser.parse_args()

            
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_path = os.path.join(args.log_path, run_name)
    if not os.path.exists(run_path):
        result_img_path, path_ckpt, path_scipts = create_log_folder(run_path)
    logging.basicConfig(filename=os.path.join(run_path, 'recording.log'), level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_cuda_info(logging)
    
    train_dataset = DataLoaderTurbVideo(args.train_path, num_frames=args.num_frames, patch_size=args.patch_size, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    val_dataset = DataLoaderTurbVideo(args.val_path, num_frames=args.num_frames, patch_size=args.patch_size, is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    model = DetiltUNet3DS(norm='LN', conv_type='dw',residual='pool', noise=0.0001).to(device)

    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)
    ######### Scheduler ###########
    epochs = args.epochs
    start_epoch = 1
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    
    ######### Resume ###########
    if args.load:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
                
        if not args.start_over:
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_lr = optimizer.param_groups[0]['lr']    
                  
        for i in range(1, start_epoch):
            scheduler.step()
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        logging.info(f'==> Resuming Training with learning rate: {new_lr}')
        print('------------------------------------------------------------------------------')

    # if len(device_ids)>1:
    #     model = nn.DataParallel(model, device_ids = device_ids)

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    # criterion_edge = losses.EdgeLoss3D()
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Start_epoch:     {start_epoch}
        Batch size:      {args.batch_size}
        Learning rate:   {new_lr}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {path_ckpt}
    ''')
    
    ######### train ###########
    best_psnr = 0

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_results_folder = OrderedDict()
        train_results_folder['psnr'] = []
        train_results_folder['ssim'] = []
        # train_results_folder['psnr_y'] = []
        # train_results_folder['ssim_y'] = []
        model.train()
        for epoch_step, data in enumerate(train_loader):
            # zero_grad
            for param in model.parameters():
                param.grad = None

            input_ = data[1].to(device)
            target = data[0].to(device)

            output_3, output_2, output = model(input_)
            loss = 0.6 * criterion_char(output, target) + \
                    0.3 * criterion_char(output_2, target) + \
                    0.1 * criterion_char(output_3, target)
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()

            for b in range(output.shape[0]):
                for i in range(output.shape[1]):
                    inp = input_[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if inp.ndim == 3:
                        inp = np.transpose(inp, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    inp = inp * 255.0
                    
                    img = output[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if img.ndim == 3:
                        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    img = img * 255.0
                    
                    img_gt = target[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    img_gt = img_gt * 255.0
                
                    train_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                    train_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))

                    # if img_gt.ndim == 3:  # RGB image
                    #     train_results_folder['psnr_y'].append(util.calculate_psnr(util.mybgr2ycbcr(img), util.mybgr2ycbcr(img_gt), border=0))
                    #     train_results_folder['ssim_y'].append(util.calculate_ssim(util.mybgr2ycbcr(img), util.mybgr2ycbcr(img_gt), border=0))
                    # else:
                    #     train_results_folder['psnr_y'] = train_results_folder['psnr']
                    #     train_results_folder['ssim_y'] = train_results_folder['ssim']
                    inp = inp.round().astype(np.uint8)  # float32 to uint8
                    img = img.round().astype(np.uint8)  # float32 to uint8
                    img_gt = img_gt.round().astype(np.uint8)  # float32 to uint8
                    
                    if epoch>=10 and epoch_step % 500 == 0:
                        pg_save = Image.fromarray(np.uint8(np.concatenate((inp, img, img_gt), axis=1))).convert('RGB')
                        pg_save.save(os.path.join(result_img_path, f'train_{epoch}_{epoch_step}_{b}_{i}.jpg'), "JPEG")

        psnr = sum(train_results_folder['psnr']) / len(train_results_folder['psnr'])
        ssim = sum(train_results_folder['ssim']) / len(train_results_folder['ssim'])
        # psnr_y = sum(train_results_folder['psnr_y']) / len(train_results_folder['psnr_y'])
        # ssim_y = sum(train_results_folder['ssim_y']) / len(train_results_folder['ssim_y'])
        logging.info('Training: Epoch {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -Loss {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}'.format(epoch, epochs, time.time()-epoch_start_time, optimizer.param_groups[0]['lr'], epoch_loss/epoch_step, psnr, ssim))

        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(path_ckpt, f"model_epoch_{epoch}.pth")) 
        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(path_ckpt, "model_last.pth")) 

        #### Evaluation ####
        if epoch % args.val_period == 0:
            test_results_folder = OrderedDict()
            test_results_folder['psnr'] = []
            test_results_folder['ssim'] = []
            # test_results_folder['psnr_y'] = []
            # test_results_folder['ssim_y'] = []
            eval_loss = 0
            model.eval()
            for s, data in enumerate(val_loader):
                input_ = data[1].to(device)
                target = data[0].to(device)
                with torch.no_grad():
                    output_3, output_2, output = model(input_)
                    loss = 0.6 * criterion_char(output, target) + \
                            0.3 * criterion_char(output_2, target) + \
                            0.1 * criterion_char(output_3, target)
                    eval_loss += loss.item()
                    
                for b in range(output.shape[0]):
                    for i in range(output.shape[1]):
                        inp = input_[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        if inp.ndim == 3:
                            inp = np.transpose(inp, (1, 2, 0))  # CHW-RGB to HWC-BGR
                        inp = inp * 255.0
                    
                        img = output[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        if img.ndim == 3:
                            img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HCW-BGR
                        img = img * 255.0
                        
                        img_gt = target[b, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        if img_gt.ndim == 3:
                            img_gt = np.transpose(img_gt, (1, 2, 0))  # CHW-RGB to HCW-BGR
                        img_gt = img_gt * 255.0

                        test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                        test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                        
                        # if img_gt.ndim == 3:  # RGB image
                        #     test_results_folder['psnr_y'].append(util.calculate_psnr(util.mybgr2ycbcr(img), util.mybgr2ycbcr(img_gt), border=0))
                        #     test_results_folder['ssim_y'].append(util.calculate_ssim(util.mybgr2ycbcr(img), util.mybgr2ycbcr(img_gt), border=0))
                        # else:
                        #     test_results_folder['psnr_y'] = test_results_folder['psnr']
                        #     test_results_folder['ssim_y'] = test_results_folder['ssim']
                            
                        inp = inp.round().astype(np.uint8)  # float32 to uint8
                        img = img.round().astype(np.uint8)  # float32 to uint8
                        img_gt = img_gt.round().astype(np.uint8)  # float32 to uint8                          
                        if epoch>=10 and s % 250 == 0:
                            pg_save = Image.fromarray(np.uint8(np.concatenate((inp, img, img_gt), axis=1))).convert('RGB')
                            pg_save.save(os.path.join(result_img_path, f'val_{epoch}_{s}_{b}_{i}.jpg'), "JPEG")
                            
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            # psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            # ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
            
            logging.info('Validation: Epoch {:d}/{:d} - Loss {:8f} - PSNR: {:.4f} dB; SSIM: {:.4f}'.
                    format(epoch, epochs, eval_loss/s, psnr, ssim))

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(path_ckpt, "model_best.pth"))
                            
        scheduler.step()    
         
if __name__ == '__main__':
    main()
