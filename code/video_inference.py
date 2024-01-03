import cv2
import torch
import argparse, os, time
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from model.TMT import TMT_MS

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
    
def get_args():
    parser = argparse.ArgumentParser(description='Video inference with overlapping patches')
    parser.add_argument('--patch_size', '-ps', dest='patch_size', type=int, default=240, help='saptial patch size')
    parser.add_argument('--temp_patch', type=int, default=12, help='temporal patch size')
    parser.add_argument('--resize_ratio', type=float, default=1.0, help='saptial resize ratio for both w and h')
    parser.add_argument('--start_frame', type=float, default=0.0, help='first frame to be processed, if < 1, it is ratio w.r.t. the entire video, if >1, it is absolute value')
    parser.add_argument('--total_frames', type=int, default=-1, help='number of total frames to be processed')
    parser.add_argument('--input_path', type=str, default=None, help='path of input video')
    parser.add_argument('--out_path', type=str, default=None, help='path of output video')
    parser.add_argument('--model_path', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--save_images', action='store_true', help='save results as images')
    parser.add_argument('--save_video', action='store_true', help='save results as video')
    parser.add_argument('--concatenate_input', action='store_true', help='concatenate input and output frames')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(0)
    
    net = TMT_MS(num_blocks=[2,3,3,4], 
                    heads=[1,2,4,8], 
                    num_refinement_blocks=2, 
                    warp_mode='none', 
                    n_frames=args.temp_patch, 
                    att_type='shuffle').cuda().train()    
    # load
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
    for name, param in net.named_parameters():
        param.requires_grad = False
    
    if args.save_images:
        frame_dir = os.path.join(os.path.split(args.out_path)[0], os.path.split(args.out_path)[1].split('.')[0])
        os.makedirs(frame_dir, exist_ok=True)
        
    # load video
    vpath = args.input_path
    print(f'processing {vpath}')
    all_frames = []
    turb_vid = cv2.VideoCapture(vpath)

    h, w = int(turb_vid.get(4)), int(turb_vid.get(3))
    fps = int(turb_vid.get(5))
    total_frames = int(turb_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(args.start_frame * total_frames) if args.start_frame < 1 else int(args.start_frame)
    turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if args.total_frames > 0:
        total_frames = args.total_frames
    test_frame_info = [{'start':0, 'range':[0,9]}]
    num_chunk = (total_frames-1) // (args.temp_patch//2)
    for i in range(1, num_chunk):
        if i == num_chunk-1:
            test_frame_info.append({'start':total_frames-args.temp_patch, 'range':[i*(args.temp_patch//2)+(args.temp_patch//4),total_frames]})
        else:
            test_frame_info.append({'start':i*(args.temp_patch//2), 
            'range':[i*(args.temp_patch//2)+(args.temp_patch//4),i*(args.temp_patch//2)+(args.temp_patch//4*3)]})

    all_frames = [turb_vid.read()[1] for i in range(total_frames)]
    turb_vid.release()
    
    if args.resize_ratio != 1:
        hh = int(h * args.resize_ratio)
        ww = int(w * args.resize_ratio)
        all_frames = [cv2.resize(f, (hh,ww)) for f in all_frames]
    else:
        hh, ww = h, w
    
    patch_size = args.patch_size
    inp_frames = []
    out_frames = []
    frame_idx = 0

    start_t = time.time()
    for i in range(num_chunk):
        print(f'{i}/{num_chunk}')
        out_range = test_frame_info[i]['range']
        in_range = [test_frame_info[i]['start'], test_frame_info[i]['start']+args.temp_patch]
        inp_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in all_frames[in_range[0]:in_range[1]]]
        inp_imgs = [TF.to_tensor(TF.resize(img, (hh, ww))) for img in inp_imgs]

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
            inp = cv2.cvtColor(restore_PIL(input_, 0, j), cv2.COLOR_RGB2BGR)
            if args.save_images:
                if args.concatenate_input:
                    oframe = np.concatenate((inp, out), axis=1)
                else:
                    oframe = out
                cv2.imwrite(os.path.join(frame_dir, '{:04d}.png'.format(frame_idx+start_frame)), oframe)
            else:
                out_frames.append(out)
                inp_frames.append(inp)
            frame_idx += 1

    print(f'total time consumption is {time.time()-start_t} s')

    if args.save_video:
        if args.concatenate_input:
            output_writer = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (ww*2, hh))
            for fid, frame in enumerate(out_frames):
                output_writer.write(np.concatenate((inp_frames[fid], frame), axis=1))
        else:
            output_writer = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (ww, hh))
            for fid, frame in enumerate(out_frames):
                output_writer.write(frame)
        output_writer.release()
