import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange


##########################################################################
## Layer Norm
class LayerNorm3D(nn.Module):
    def __init__(self, dim, elementwise_affine=True, bias=True):
        super(LayerNorm3D, self).__init__()
        self.LN = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)
    
    def to_3d(self, x):
        return rearrange(x, 'b c t h w -> b (t h w) c')

    def to_5d(self,x,t,h,w):
        return rearrange(x, 'b (t h w) c -> b c t h w', t=t, h=h,w=w)

    def forward(self, x):
        t, h, w = x.shape[-3:]
        return self.to_5d(self.LN(self.to_3d(x)), t, h, w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = LayerNorm3D(dim, bias=False)
        else:
            self.body =LayerNorm3D(dim)

    def forward(self, x):
        return self.body(x)


def TiltWarp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, applyonfeature=True):
    """
    Args:
        x (Tensor): Tensor with size (b, n, c, h, w) -> (b*n, c, h, w).
        flow (Tensor): Tensor with size (b, 2, n, h, w) -> (b*n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.
    Returns:
        Tensor: Warped image or feature map.
    """
    if applyonfeature:
        _, c, n, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
    else: 
        _, n, c, h, w = x.size()
        x = x.reshape((-1, c, h, w))
    flow = flow.permute(0,2,3,4,1).reshape((-1, h, w, 2))
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), 
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    vgrid = grid + flow

    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    if applyonfeature:
        output = rearrange(x, '(b t) c h w -> b c t h w', t=n)
    else:
        output = output.reshape((-1, n, c, h, w))
    return output

class DWconv3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
                    bias=True, padding_mode='zeros', device=None, dtype=None):
        super(DWconv3D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
                                groups=in_channels,
                                bias=bias, padding_mode=padding_mode, device=device, dtype=dtype),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)


class Conv3D_Block(nn.Module):
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, norm=nn.BatchNorm3d, conv_type='normal', residual=None):
        super(Conv3D_Block, self).__init__()

        if conv_type == 'normal':
            conv3d = nn.Conv3d
        elif conv_type == 'dw':
            conv3d = DWconv3D
            
        self.conv1 = nn.Sequential(
            conv3d(inp_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            norm(out_feat),
            nn.LeakyReLU())

        self.conv2 = nn.Sequential(
            conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            norm(out_feat),
            nn.LeakyReLU())

        self.residual = residual

        if self.residual == 'conv':
            self.residual_upsampler = conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        res = x
        if self.residual == 'conv':
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class FeatureAlign(nn.Module):
    def __init__(self, inp_feat, embed_feat):
        super(FeatureAlign, self).__init__()

        self.conv_blk = Conv3D_Block(inp_feat, embed_feat, norm=LayerNorm3D, conv_type='dw', residual='conv')
        self.out = nn.Conv3d(embed_feat, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        flow = self.out(self.conv_blk(x))
        out = TiltWarp(x, flow)
        return out


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, flow_dim_ratio, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.dim = dim
        self.flow_dim = int(dim * flow_dim_ratio)
        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim+self.flow_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        if self.flow_dim == 0:
            x = self.project_out(x)
            return x
        else:
            x, f = self.project_out(x).split([self.dim, self.flow_dim], dim=1)
            return x, f

##########################################################################
## channel-temporal shuffle attention
class AttentionCTSF(nn.Module):
    def __init__(self, dim, num_heads, bias, n_frames=10):
        super(AttentionCTSF, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        shuffle_g = 8
        # shuffle_g = 16
        self.get_qkv = nn.Sequential(nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias),
                            nn.Conv3d(dim*3, dim*3, kernel_size=(1,3,3), stride=1, padding=(0,1,1), groups=dim*3, bias=bias),
                            Rearrange('b (c1 c2) t h w -> b c2 h w (c1 t)', c1=shuffle_g),
                            nn.Linear(n_frames*shuffle_g, n_frames*shuffle_g),
                            Rearrange('b c2 h w (c1 t) -> b (c2 c1 t) h w', c1=shuffle_g))
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,t,h,w = x.shape

        qkv = self.get_qkv(x)
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c t) h w -> b head (c t) (h w)', head=self.num_heads, t=t)
        k = rearrange(k, 'b (head c t) h w -> b head (c t) (h w)', head=self.num_heads, t=t)
        v = rearrange(v, 'b (head c t) h w -> b head (c t) (h w)', head=self.num_heads, t=t)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head (c t) (h w) -> b (head c) t h w', head=self.num_heads, t=t, h=h, w=w)
        out = self.project_out(out)
        return out   
        
##########################################################################
## channel-temporal shuffle attention
class AttentionCTSFNew(nn.Module):
    def __init__(self, dim, num_heads, bias, n_frames=10):
        super(AttentionCTSFNew, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.PE = nn.Parameter(torch.zeros(1, dim, n_frames, 1, 1))
        shuffle_g = 8
        # shuffle_g = 16
        self.get_qkv = nn.Sequential(nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias),
                            nn.Conv3d(dim*3, dim*3, kernel_size=(1,3,3), stride=1, padding=(0,1,1), groups=dim*3, bias=bias),
                            Rearrange('b (c1 c2) t h w -> b c2 h w (c1 t)', c1=shuffle_g),
                            nn.Linear(n_frames*shuffle_g, n_frames*shuffle_g),
                            Rearrange('b c2 h w (c1 t) -> b (c2 c1 t) h w', c1=shuffle_g))
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,t,h,w = x.shape
        
        qkv = self.get_qkv(x + self.PE)
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c t) h w -> b head (c t) (h w)', head=self.num_heads, t=t)
        k = rearrange(k, 'b (head c t) h w -> b head (c t) (h w)', head=self.num_heads, t=t)
        v = rearrange(v, 'b (head c t) h w -> b head (c t) (h w)', head=self.num_heads, t=t)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head (c t) (h w) -> b (head c) t h w', head=self.num_heads, t=t, h=h, w=w)
        out = self.project_out(out)
        return out   
        
        
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, att, ffn_expansion_factor, bias, LN_bias, flow_dim_ratio, att_ckpt, ffn_ckpt, n_frames=10):
        super(TransformerBlock, self).__init__()
        self.flow_dim_ratio = flow_dim_ratio
        self.att_ckpt = att_ckpt
        self.ffn_ckpt = ffn_ckpt
        
        self.norm1 = LayerNorm3D(dim, bias=LN_bias)
        if att == 'sep':
            self.attn = AttentionCTS(dim, num_heads, bias, n_frames)
        elif att == 'shuffle':
            self.attn = AttentionCTSF(dim, num_heads, bias, n_frames)
        elif att == 'simple':
            self.attn = Simple(dim, bias, n_frames)
        self.norm2 = LayerNorm3D(dim, bias=LN_bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, flow_dim_ratio, bias)

    def forward(self, x):
        x = self.norm1(x)
        if self.att_ckpt:
            x = x + checkpoint.checkpoint(self.attn, x)
        else:
            x = x + self.attn(x)
        
        x = self.norm2(x)
        if self.ffn_ckpt:
            o = checkpoint.checkpoint(self.ffn, x)
        else:
            o = self.ffn(x)
        
        if self.flow_dim_ratio > 0:
            return x + o[0], o[1]
        else:
            return x + o


##########################################################################
## Preprocessing with 3x7x7 Conv
class Preprocessing(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(Preprocessing, self).__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=(3,7,7), stride=1, padding=(1,3,3), bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(dim_in, dim_out//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  Rearrange('b c t (h rh) (w rw) -> b (c rh rw) t h w', rh=2, rw=2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  Rearrange('b (c rh rw) t h w -> b c t (h rh) (w rw)', rh=2, rw=2))

    def forward(self, x):
        return self.body(x)


def make_level_blk(dim, num_tb, nhead, att, ffn, bias, LN_bias, 
                flow_dim_ratio=0, att_ckpt=False, ffn_ckpt=False, n_frames=10, self_align=True):
    module = []
    for i in range(num_tb-1):
        module.append(TransformerBlock(dim=dim, 
                                        num_heads=nhead, 
                                        att=att,
                                        ffn_expansion_factor=ffn, 
                                        bias=bias, 
                                        LN_bias=LN_bias,
                                        flow_dim_ratio=0, 
                                        att_ckpt=att_ckpt, 
                                        ffn_ckpt=ffn_ckpt, 
                                        n_frames=n_frames))
        if i == int(num_tb/2) and self_align:
            module.append(FeatureAlign(dim, dim))

    module.append(TransformerBlock(dim=dim, 
                            num_heads=nhead, 
                            att=att,
                            ffn_expansion_factor=ffn, 
                            bias=bias, 
                            LN_bias=LN_bias,
                            flow_dim_ratio=flow_dim_ratio, 
                            att_ckpt=att_ckpt, 
                            ffn_ckpt=ffn_ckpt, 
                            n_frames=n_frames))
    return module

##########################################################################
class TMT_MS(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LN_bias = True,
        warp_mode = 'none',
        n_frames = 10,
        att_type = 'shuffle',
        out_residual = True,
        att_ckpt = False,
        ffn_ckpt = False
    ):

        super(TMT_MS, self).__init__()
        if warp_mode == 'enc':
            align = [True, True, True, False, False, False, False]
        elif warp_mode == 'dec':
            align = [False, False, False, True, True, True, True]
        elif warp_mode == 'all':
            align = [True, True, True, True, True, True, True]
        else:
            align = [False, False, False, False, False, False, False]
        
        self.out_residual = out_residual
        
        self.getFeature1 = Preprocessing(inp_channels, dim)
        self.getFeature2 = Preprocessing(inp_channels, dim)
        self.getFeature3 = Preprocessing(inp_channels, dim)

        self.encode_l1 = nn.Sequential(*make_level_blk(dim=dim, num_tb=num_blocks[0], nhead=heads[0], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[0]))
        
        self.down1_2 = Downsample(dim, int(dim*2**1)-dim) ## From Level 1 to Level 2
        self.encode_l2 = nn.Sequential(*make_level_blk(dim=int(dim*2**1), num_tb=num_blocks[1], nhead=heads[1], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[1]))

        
        self.down2_3 = Downsample(int(dim*2**1), int(dim*2**2)-dim) ## From Level 2 to Level 3
        self.encode_l3 = nn.Sequential(*make_level_blk(dim=int(dim*2**2), num_tb=num_blocks[2], nhead=heads[2], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[2]))

        self.down3_4 = Downsample(int(dim*2**2), int(dim*2**3)) ## From Level 3 to Level 4
        self.embedding = nn.Sequential(*make_level_blk(dim=int(dim*2**3), num_tb=num_blocks[3], nhead=heads[3], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[3]))     
                                                   
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_l3 = nn.Conv3d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decode_l3 = nn.Sequential(*make_level_blk(dim=int(dim*2**2), num_tb=num_blocks[2], nhead=heads[2], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[4])) 


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_l2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decode_l2 = nn.Sequential(*make_level_blk(dim=int(dim*2**1), num_tb=num_blocks[1], nhead=heads[1], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[5])) 
        
        self.up2_1 = Upsample(int(dim*2**1))

        self.decode_l1 = nn.Sequential(*make_level_blk(dim=int(dim*2**1), num_tb=num_blocks[0], nhead=heads[0], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[6])) 

        self.refinement = nn.Sequential(*make_level_blk(dim=int(dim*2**1), num_tb=num_refinement_blocks, nhead=heads[0], att=att_type,
                                                ffn=ffn_expansion_factor, bias=bias, LN_bias=LN_bias, att_ckpt=att_ckpt, 
                                                ffn_ckpt=ffn_ckpt, n_frames=n_frames, self_align=align[6]))         
            
        self.output = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        b,c,t,h,w = inp_img.shape
        inp_img2 = F.interpolate(inp_img, size=(t, h//2, w//2), mode='trilinear', align_corners=False)
        inp_img3 = F.interpolate(inp_img, size=(t, h//4, w//4), mode='trilinear', align_corners=False)
        inp_enc_l1 = self.getFeature1(inp_img)
        inp_enc_l2 = self.getFeature2(inp_img2)
        inp_enc_l3 = self.getFeature3(inp_img3)
        
        out_enc_l1 = self.encode_l1(inp_enc_l1)
        out_enc_l2 = self.encode_l2(torch.cat([inp_enc_l2, self.down1_2(out_enc_l1)], 1))
        out_enc_l3 = self.encode_l3(torch.cat([inp_enc_l3, self.down2_3(out_enc_l2)], 1))

        inp_enc_l4 = self.down3_4(out_enc_l3)
        embedding = self.embedding(inp_enc_l4)

        inp_dec_l3 = self.up4_3(embedding)
        inp_dec_l3 = torch.cat([inp_dec_l3, out_enc_l3], 1)
        inp_dec_l3 = self.reduce_chan_l3(inp_dec_l3)
        out_dec_l3 = self.decode_l3(inp_dec_l3) 

        inp_dec_l2 = self.up3_2(out_dec_l3)
        inp_dec_l2 = torch.cat([inp_dec_l2, out_enc_l2], 1)
        inp_dec_l2 = self.reduce_chan_l2(inp_dec_l2)
        out_dec_l2 = self.decode_l2(inp_dec_l2) 

        inp_dec_l1 = self.up2_1(out_dec_l2)
        inp_dec_l1 = torch.cat([inp_dec_l1, out_enc_l1], 1)
        out_dec_l1 = self.decode_l1(inp_dec_l1)
        
        out_dec_l1 = self.refinement(out_dec_l1)
        
        if self.out_residual:
            out = self.output(out_dec_l1) + inp_img
        else:
            out = self.output(out_dec_l1)
        return out


def restore_PIL(tensor, b, fidx):
    img = tensor[b, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return img
            
if __name__ == '__main__':
    from torchsummary import summary
    from PIL import Image
    import torchvision.transforms.functional as TF
    from UNet3d import DetiltUNet3D
    import cv2
    import time
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    torch.cuda.set_device(0)
    net = TMT_MS(num_blocks=[2,3,4,4], 
                    heads=[2,4,6,8], 
                    num_refinement_blocks=2, 
                    warp_mode='none', 
                    n_frames=12, 
                    att_type='shuffle',
                    att_ckpt=False,
                    ffn_ckpt=False).cuda().train()
    # torch.save(net.state_dict(), 'model_shuffle.pth')
    # summary(net, (3,10,128,128))
    # summary(net, (3,20,128,128))
    # 22.9528 1584~12f 
    with torch.no_grad():
        s = time.time()
        for i in range(1):
            inputs = torch.randn(1,3,12,208,208).cuda()
            print('{:>16s} : {:<.4f} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))
            flops = FlopCountAnalysis(net, inputs)
            print(flop_count_table(flops))
            print(flops.total())
            # net(inputs)
        print(time.time()-s)
