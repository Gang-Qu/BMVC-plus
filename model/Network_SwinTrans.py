import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import torch.nn.init as init
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from utils import *
from timm.models.vision_transformer import Mlp, PatchEmbed
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from mamba_ssm import Mamba
from timm.models.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
import scipy.io as scio



class MambaLayer(nn.Module):
    def __init__(self, dim, dim_1, dim_2, d_state=16, d_conv=4, expand=2, channel_token=False, mode_temp=3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim_1)
        self.norm3 = nn.LayerNorm(dim_2)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba= True,
        )
        self.mamba2 = Mamba(
            d_model=dim_1,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba= True,
        )
        self.mamba3 = Mamba(
            d_model=dim_2,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba= True,
        )
        self.channel_token = channel_token  ## whether to use channel as tokens
        self.mode_temp = mode_temp

    def forward_patch_token(self, x, r=False):

        mode_list = []
        mode_temp = [0, 1, 2]
        x_1 = x
        for mode in mode_temp:

            three_d_tensors = torch.unbind(x_1, dim=0)

            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                if mode == 0:
                    tensor = tensor
                elif mode == 1:
                    tensor = tensor.permute(1, 0, 2)
                elif mode == 2:
                    tensor = tensor.permute(2, 0, 1)
                tensor_list.append(tensor)

            reconstructed_x = torch.stack(tensor_list, dim=0)
            x = reconstructed_x.unsqueeze(2)
            B, C, T, H, W = x.shape

            B, d_model = x.shape[:2]
            d_model = C
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]
            x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
            if mode == 0:
                x_norm = self.norm(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba(x_norm)
            elif mode == 1:
                x_norm = self.norm2(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba2(x_norm)

            elif mode == 2:
                x_norm = self.norm3(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba3(x_norm)

            out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims).contiguous()

            out = out.squeeze(2)

            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor_out_last in enumerate(three_d_tensors_out):
                if mode == 0:
                    tensor_out = tensor_out_last
                elif mode == 1:
                    tensor_out = tensor_out_last.permute(1, 0, 2)
                elif mode == 2:
                    tensor_out = tensor_out_last.permute(1, 2, 0)
                tensor_list_out.append(tensor_out)

            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)

            mode_list.append(reconstructed_x_out)

        mean = (self.alpha * mode_list[0] + self.beta * mode_list[1] + self.gamma * mode_list[2])

        return mean

    def forward_channel_token(self, x, r=False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    def forward(self, x, r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out


def EuclideanProj(X, Y, H, W, HT, WT, mu):
    b, c, _, _ = X.shape
    Delta_Y = Y - torch.matmul(torch.matmul(H.repeat((b, c, 1, 1)), X), WT.repeat((b, c, 1, 1)))
    Delta_X = torch.matmul(torch.matmul(HT.repeat((b, c, 1, 1)), Delta_Y), W.repeat((b, c, 1, 1)))
    Z = X + torch.div(Delta_X, mu + 1)
    return Z


def GAPProj(X, Y, H, H_s, mu):
    Delta_Y = Y - A(X, H)
    Z = X + torch.div(At(torch.div(Delta_Y, H_s), H), mu + 1)
    return Z


def ImageDecompose(Image, cr1, cr2):
    b, c, h, w = Image.shape
    blocks = torch.zeros((b, cr1 * cr2, h // cr1, w // cr2), device=Image.device, dtype=Image.dtype)
    for q in range(cr1):
        row_idx = slice(q, h, cr1)
        for g in range(cr2):
            col_idx = slice(g, w, cr2) if q % 2 == 0 else slice(cr2 - g - 1, w, cr2)
            blocks[:, q * cr2 + g] = Image[:, 0, row_idx, col_idx]  # 只取第一个通道
    return blocks


def ImageCombine(Image, cr1, cr2):
    b, c, h, w = Image.shape
    H, W = cr1 * h, cr2 * w
    merged = torch.zeros((b, 1, H, W), device=Image.device, dtype=Image.dtype)
    for q in range(cr1):
        row_idx = slice(q, H, cr1)
        for g in range(cr2):
            col_idx = slice(g, W, cr2) if q % 2 == 0 else slice(cr2 - g - 1, W, cr2)
            merged[:, :, row_idx, col_idx] = Image[:, q * cr2 + g]
    return merged


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ConvWithCA(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            ChannelAttention(dim),
            nn.Conv2d(dim, dim_out, 3, 1, 1),
        )

    def forward(self, x):
        '''
        x: B (H W) C
        '''
        x = self.cab(x)
        return x


class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor=2.66
                 ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x


class ChannelAttention2D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, (dim // 2) * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        C = C // 2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4,
                                                                                         1)  ## (1, 4) channel weight  (4, 1) channel attention
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x


class DCCABlock(nn.Module):
    def __init__(self, dim, meas_size):
        super().__init__()
        self.scb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

        self.mb = MambaLayer(dim=dim,
                             dim_1=meas_size[0],
                             dim_2=meas_size[1],
                             d_state=16,
                             d_conv=4,
                             expand=2,
                             channel_token=False,
                             mode_temp=3)
        self.csab = ChannelAttention2D(dim, num_heads=4)
        self.trans_block1 = SwinBlock(dim, head_dim=32, window_size=4, ffn_ratio=2, type='W')
        self.trans_block2 = SwinBlock(dim, head_dim=32, window_size=4, ffn_ratio=2, type='SW')
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        csab_in = einops.rearrange(x, "b c h w->  (h w) b c")
        csab_out = self.csab(csab_in)
        csab_out = einops.rearrange(csab_out, " (h w) b c->b c h w", h=h, w=w)

        x = x + csab_out
        x1 = self.trans_block1(x)
        trans_out = self.trans_block2(x1)
        x = x + trans_out
        ffn_out = x + self.ffn(x)

        return ffn_out


def RelativePosition(window_size):
    coords_d = torch.arange(window_size[0])
    coords_h = torch.arange(window_size[1])
    coords_w = torch.arange(window_size[2])
    coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
    relative_coords1 = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
    relative_coords = relative_coords1.permute(1, 2, 0).contiguous()  #
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 2] += window_size[2] - 1

    relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
    relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
    relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
    return relative_position_index


class WMSA(nn.Module):
    """
    Window Multi-head Self-Attention (WMSA) module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.qkv = nn.Linear(self.input_dim, 3 * self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        # relative position encoding
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        trunc_normal_(self.relative_position_params, std=.02)
        relative_position_index = RelativePosition(window_size=[1, self.window_size, self.window_size])
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        h, w = x.shape[1:-1]
        assert h % self.window_size == 0 and w % self.window_size == 0, 'Input cannot be divided into spatial windows.'
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (w1 p1) (w2 p2) (threeh c) -> threeh b (w1 w2) (p1 p2) c', p1=self.window_size,
                            p2=self.window_size, c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative opsition bias
        sim = sim + rearrange(self.relative_position_params[self.relative_position_index.reshape(-1)],
                              '(p q) h -> h 1 1 p q', p=self.window_size ** 2, q=self.window_size ** 2)
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(int(h / self.window_size), int(w / self.window_size), self.window_size,
                                           shift=self.window_size // 2).to(sim.device)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        atten = F.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', atten, v)
        output = rearrange(output, 'h b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) (h c)', w1=int(h / self.window_size),
                           p1=self.window_size)
        output = self.linear(output)

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                 dims=(1, 2))
        return output

class SwinBlock(nn.Module):
    def __init__(self, dim, head_dim, window_size, ffn_ratio=2, type='W'):
        """ Our Shiwin Transformer Block
        """
        super().__init__()
        assert type in ['W', 'SW']
        self.msa = WMSA(dim, dim, head_dim, window_size, type)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_ratio * dim, 1, 1, 0, bias=True),
            # nn.GELU(),
            nn.Conv2d(ffn_ratio * dim, ffn_ratio * dim, 3, 1, 1, groups=ffn_ratio * dim, bias=True),
            nn.GELU(),
            nn.Conv2d(ffn_ratio * dim, dim, 1, 1, 0, bias=True),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x):
        x = Rearrange('b c h w -> b h w c')(x)
        x = x + self.msa(self.ln1(x))
        x = x + self.ffn(self.ln2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = Rearrange('b h w c -> b c h w')(x)
        return x


class ResDNetBlock(nn.Module):
    def __init__(self, dim, group_num, meas_size):
        super().__init__()
        self.dcca_list = nn.ModuleList()
        self.group_num = group_num
        group_dim = dim // group_num
        self.dense_conv = nn.ModuleList()
        for i in range(group_num):
            self.dcca_list.append(DCCABlock(group_dim, meas_size))
            if i > 0:
                self.dense_conv.append(
                    nn.Sequential(
                        nn.Conv2d(group_dim * (i + 1), group_dim, 1),
                        nn.LeakyReLU(inplace=True)
                    )
                )
        self.last_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        input_list = torch.chunk(x, chunks=self.group_num, dim=1)
        cf_in = input_list[0]
        out_list = []
        cf_out = self.dcca_list[0](cf_in)
        out_list.append(cf_out)
        for i in range(1, self.group_num):
            in_list = out_list.copy()
            in_list.append(input_list[i])
            cf_in = torch.cat(in_list, dim=1)
            cf_in = self.dense_conv[i - 1](cf_in)
            cf_out = self.dcca_list[i](cf_in)
            out_list.append(cf_out)
        out = torch.cat(out_list, dim=1)
        out = self.last_conv(out)
        out = x + out
        return out


class BasicBlock(nn.Module):
    def __init__(self, meas_size, color_channels=1, width=64, middle_blk_num=6, enc_blk_nums=[1, 2, 4],
                 dec_blk_nums=[1, 1, 1],
                 first_stage=False):
        super(BasicBlock, self).__init__()
        self.first_stage = first_stage
        self.embedding = nn.Conv2d(in_channels=color_channels, out_channels=width, kernel_size=3, padding=1, stride=1,
                                   groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        if not first_stage:
            self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mapping = nn.Conv2d(in_channels=width, out_channels=color_channels, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            meas_size = meas_size

            if not first_stage:
                self.convs1.append(
                    # ConvWithCA(chan * 2, chan)
                    nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
                )
            self.encoders.append(
                nn.Sequential(
                    *[ResDNetBlock(dim=chan, group_num=1, meas_size=meas_size) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)

            )
            meas_size = [int(num / 2) for num in meas_size]
            chan = chan * 2
        if not first_stage:
            self.middle_blks = nn.Sequential(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False),
                # ConvWithCA(chan * 2, chan),
                *[ResDNetBlock(dim=chan, group_num=1, meas_size=meas_size) for _ in range(middle_blk_num)])
        else:
            self.middle_blks = nn.Sequential(
                *[ResDNetBlock(dim=chan, group_num=1, meas_size=meas_size) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            meas_size = [int(num * 2) for num in meas_size]

            self.convs2.append(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False)

            )
            self.decoders.append(
                nn.Sequential(
                    *[ResDNetBlock(dim=chan, group_num=1, meas_size=meas_size) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, last_decs=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.embedding(inp)

        encs = []
        decs = []
        if not self.first_stage:
            for encoder, down, conv1, last_dec in zip(self.encoders, self.downs, self.convs1, last_decs[::-1]):
                x = conv1(torch.cat([x, last_dec], dim=1))
                x = encoder(x)
                encs.append(x)
                x = down(x)
        else:
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encs.append(x)
                x = down(x)

        if not self.first_stage:
            x = self.middle_blks(torch.cat([x, last_decs[::-1][2]], dim=1))
        else:
            x = self.middle_blks(x)
        decs.append(x)
        for decoder, up, conv2, enc_skip in zip(self.decoders, self.ups, self.convs2, encs[::-1]):
            x = up(x)
            x = conv2(torch.cat([x, enc_skip], dim=1))
            x = decoder(x)
            decs.append(x)
        x = self.mapping(x)
        x = x + inp
        return x[:, :, :H, :W], decs  ## decs_atten

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class DualBMVC(nn.Module):
    def __init__(self, imag_size, cr, meas_size, channels=64, mid_blocks=6, enc_blocks=[1, 2, 4],
                 dec_blocks=[1, 1, 1], stages=7, matrix_train=True, only_test=False):
        super().__init__()
        self.stages = stages
        self.imag_size = imag_size
        self.only_test = only_test
        self.cr1 = cr[0]
        self.cr2 = cr[1]
        self.channel = self.cr1 * self.cr2
        # self.H = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, imag_size[0], imag_size[1])), requires_grad=True)
        self.H = nn.Parameter(nn.init.uniform_(torch.Tensor(1, 1, imag_size[0], imag_size[1]), a=0, b=1),
                              requires_grad=True)
        self.mu = nn.Parameter(torch.Tensor([0.001]).repeat(stages))
        self.denoisers = nn.ModuleList([])
        self.denoisers.append(
            BasicBlock(meas_size=meas_size, color_channels=self.channel, width=channels, middle_blk_num=mid_blocks,
                       enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks, first_stage=True))
        for i in range(stages - 1):
            self.denoisers.append(
                BasicBlock(meas_size=meas_size, color_channels=self.channel, width=channels, middle_blk_num=mid_blocks,
                           enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks))

    def forward(self, X):
        """
        :input X: [b,256,256]
        """
        X_list = []
        X = X.unsqueeze(1)
        b, c, h, w = X.shape
        H0 = self.H.repeat((b, 1, 1, 1))
        XX = ImageDecompose(X[:,:,:self.imag_size[0], :self.imag_size[1]], self.cr1, self.cr2)
        H = ImageDecompose(H0, self.cr1, self.cr2)
        H_s = torch.sum(H, 1)
        Y0 = torch.mul(H, XX)
        Y = torch.sum(Y0, dim=1)
        X = At(Y, H)

        for i in range(self.stages):
            mu = self.mu[i]
            Z = GAPProj(X, Y, H, H_s, mu)
            if i == 0:
                X, features = self.denoisers[i](Z)
            else:
                X, features = self.denoisers[i](Z, features)
        X0 = ImageCombine(X, self.cr1, self.cr2)
        X_list.append(X0)
        return X_list
