# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import time
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath, Block
from timm.models.swin_transformer_v2 import SwinTransformerV2
import numpy as np
import math
import random

from timm.models.layers import to_2tuple, to_3tuple
from pos_embed import get_3d_sincos_pos_embed
import VQGAN


################# Layers ########################


class identical_converter(nn.Module):
    def __init__(self, in_dim, out_dim=None, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim else in_dim
        assert self.in_dim == self.out_dim
        self.fc1 = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        return x


class linear_converter(nn.Module):
    def __init__(self, in_dim, out_dim=None, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim else in_dim
        self.fc1 = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


class mlp_converter(nn.Module):
    def __init__(self, in_dim, out_dim=None, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim else in_dim
        self.fc1 = Mlp(self.in_dim, out_features=self.out_dim, drop=0)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Maxout_converter(nn.Module):
    def __init__(self, in_dim, out_dim=None, num_pieces=2, **kwargs):
        super(Maxout_converter, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim else in_dim

        self.fc1 = nn.Linear(self.in_dim, self.out_dim * num_pieces)
        self.num_pieces = num_pieces

    def forward(self, x):
        x = F.max_pool1d(self.fc1(x), self.num_pieces)

        return x


class upsampler(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, k=3, upsize=(2, 2)):
        super(upsampler, self).__init__()
        upsize = to_2tuple(upsize)
        self.up = nn.ConvTranspose2d(dim_in, dim_in, stride=upsize, kernel_size=upsize, padding=0, output_padding=0)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=k, stride=1, padding=int(np.ceil((k - 1) / 2)), bias=True)

        self.relu = nn.SiLU(True)
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        x = self.bn(x)

        self.relu(x)

        return x


class PatchEmbed3D(nn.Module):
    """3D Volume to Patch Embedding"""

    def __init__(
            self,
            img_size=(40, 1000, 160),
            patch_size=(4, 500, 8),
            embed_dim=512,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_size_t = (1, patch_size[1], 1)
        self.patch_size_s = (patch_size[0], 1, patch_size[2])
        self.grid_size = tuple(img_size[i] // patch_size[i] for i in range(3))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj_time = nn.Conv3d(1, embed_dim // patch_size[0], kernel_size=self.patch_size_t, stride=self.patch_size_t, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Conv3d(embed_dim // patch_size[0], embed_dim, kernel_size=self.patch_size_s, stride=self.patch_size_s, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, D, H, W = x.shape
        assert D == self.img_size[0], f"Input depth ({D}) doesn't match model ({self.img_size[0]})."
        assert H == self.img_size[1], f"Input height ({H}) doesn't match model ({self.img_size[1]})."
        assert W == self.img_size[2], f"Input width ({W}) doesn't match model ({self.img_size[2]})."

        x = self.proj_time(x.unsqueeze(1))  # [B, embed_dim, D', H', W']
        x = self.act(x)
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, embed_dim, N] -> [B, N, embed_dim]
        x = self.norm(x)
        return x


################# Blocks ########################

class simfwi_Encoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(40, 1000, 160), patch_size=(5, 500, 8),
                 embed_dim=528, depth=6, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, transformer_block=Block, **kwarg):
        super().__init__()

        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)

        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.patch_embed = PatchEmbed3D(img_size, patch_size, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        assert self.num_patch[0] * self.num_patch[1] * self.num_patch[2] == self.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [transformer_block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patch, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)

        return latent


class swin_Encoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=(40, 1000, 160), patch_size=(5, 100, 8),
                 embed_dim=768, depth=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=5, drop_path=0.,
                 norm_layer=nn.LayerNorm, **kwarg):
        super().__init__()

        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)

        num_layers = len(depth) - 1
        self.num_patch = (img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = (self.num_patch[0] // 2 ** num_layers) * (self.num_patch[1] // 2 ** num_layers)

        embed_dim = embed_dim // 2 ** num_layers

        self.blocks = SwinTransformerV2(img_size=img_size[1:], patch_size=patch_size[1:], in_chans=img_size[0], num_classes=0, global_pool='',
                                        embed_dim=embed_dim, depths=depth, num_heads=num_heads, window_size=window_size,
                                        norm_layer=norm_layer, drop_path=drop_path
                                        )

    def forward(self, imgs):
        latent = self.blocks(imgs)

        if latent.dim() == 4:
            B, H, W, C = latent.shape
            latent = latent.reshape(B, H * W, C)

        return latent


class simfwi_Decoder_cnn(nn.Module):
    """ with neck linear
    """

    def __init__(self, img_size=(400, 160), in_chans=1, embed_dim=512, decoder_embed_dim=512, drop_path=0., num_res_blocks=1, ch_mult=(1, 2, 4),
                 curr_res=(25, 10), **kwarg):
        super().__init__()

        self.conv_in = torch.nn.Conv2d(embed_dim,
                                       decoder_embed_dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.decoder = VQGAN.UpsampleDecoder(decoder_embed_dim, in_chans, num_res_blocks=num_res_blocks, resolution=img_size,
                                             ch_mult=ch_mult, dropout=drop_path, curr_res=curr_res)

    def forward(self, latent):
        latent = self.conv_in(latent)
        pred = self.decoder(latent)

        return pred


################# Inverse ########################

class pde_net(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, simg_size=(40, 1000, 160), spatch_size=(5, 500, 8), sembed_dim=528, sdepth=5,
                 vimg_size=(400, 160), vpatch_size=(16, 16), vin_chans=1,
                 num_heads=16, decoder_embed_dim=512, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, window_size=5,
                 encoder_block=simfwi_Encoder, decoder_block=simfwi_Decoder_cnn, transformer_block=Block, num_res_blocks=0, sample_source=False, **kwargs):
        super().__init__()
        simg_size = to_3tuple(simg_size)
        spatch_size = to_3tuple(spatch_size)
        vimg_size = to_2tuple(vimg_size)
        vpatch_size = to_2tuple(vpatch_size)

        self.sample_source = sample_source
        if sample_source:
            assert simg_size[0] == 3

        self.vimg_size = vimg_size
        self.vnum_patch = (int(vimg_size[0] // vpatch_size[0]), int(vimg_size[1] // vpatch_size[1]))

        self.seismic_encoder = encoder_block(img_size=simg_size, patch_size=spatch_size,
                                             embed_dim=sembed_dim, depth=sdepth, num_heads=num_heads,
                                             mlp_ratio=mlp_ratio, norm_layer=norm_layer, transformer_block=transformer_block, drop_path=drop_path,
                                             window_size=window_size)

        snum_patches = self.seismic_encoder.num_patches
        vnum_patches = int((vimg_size[0] // vpatch_size[0]) * (vimg_size[1] // vpatch_size[1]))
        self.token_mixer = Mlp(in_features=snum_patches, out_features=vnum_patches)

        num_up = (math.ceil(math.log2(vpatch_size[0])), math.ceil(math.log2(vpatch_size[1])))
        ch_mult = []
        for i in range(max(num_up) + 1):
            ch = [2, 2]
            if i >= num_up[0]:
                ch[0] = 1
            if i >= num_up[1]:
                ch[1] = 1
            ch_mult.append(ch)
        self.velocity_decoder = decoder_block(img_size=vimg_size, in_chans=vin_chans, embed_dim=sembed_dim,
                                              decoder_embed_dim=decoder_embed_dim, drop_path=drop_path, num_res_blocks=num_res_blocks,
                                              ch_mult=ch_mult, curr_res=self.vnum_patch)

    def forward_converter(self, seismic_latent):
        B, L, C = seismic_latent.shape
        seismic_latent = seismic_latent.permute(0, 2, 1)
        s2v_latent = self.token_mixer(seismic_latent).reshape(B, C, self.vnum_patch[0], self.vnum_patch[1])

        return s2v_latent

    def forward_unpatchify(self, x):
        h, w = x.shape[-2:]
        i = int(round((h - self.vimg_size[0]) / 2.))
        j = int(round((w - self.vimg_size[1]) / 2.))

        x = x[..., i:(i + self.vimg_size[0]), j:(j + self.vimg_size[1])]

        return x

    def forward(self, seismic):
        if self.sample_source:
            seismic = seismic[:, :19:9]
        seismic_latent = self.seismic_encoder(seismic)  # B, L, C

        s2v_latent = self.forward_converter(seismic_latent)  # B, C, H', W'

        s2v_pred = self.velocity_decoder(s2v_latent)

        s2v_pred = self.forward_unpatchify(s2v_pred)

        return s2v_pred


################# Define Network ########################

def prostate_inverse(simg_size=(40, 1000, 160), spatch_size=(5, 500, 8), sembed_dim=528, sdepth=6,
                     vimg_size=(400, 160), vpatch_size=(16, 16), vin_chans=1,
                     num_heads=16, decoder_embed_dim=512, drop_path=0.,
                     mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     encoder_block=simfwi_Encoder, decoder_block=simfwi_Decoder_cnn, transformer_block=Block, num_res_blocks=0, **kwargs):
    model = pde_net(simg_size=simg_size, spatch_size=spatch_size, sembed_dim=sembed_dim, sdepth=sdepth,
                    vimg_size=vimg_size, vpatch_size=vpatch_size, vin_chans=vin_chans,
                    num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, drop_path=drop_path,
                    mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                    encoder_block=encoder_block, decoder_block=decoder_block, transformer_block=transformer_block, num_res_blocks=num_res_blocks)
    return model


def prostate_inverse_swin_2d(simg_size=(40, 1000, 160), spatch_size=(5, 25, 4), sembed_dim=96, sdepth=(2, 2, 6, 2),
                             vimg_size=(400, 160), vpatch_size=(16, 16), vin_chans=1,
                             num_heads=(3, 6, 12, 24), decoder_embed_dim=512, drop_path=0.,
                             mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=5,
                             encoder_block=swin_Encoder, decoder_block=simfwi_Decoder_cnn, transformer_block=Block, num_res_blocks=0, **kwargs):
    sembed_dim = sembed_dim * 2**(len(sdepth)-1)
    model = pde_net(simg_size=simg_size, spatch_size=spatch_size, sembed_dim=sembed_dim, sdepth=sdepth,
                    vimg_size=vimg_size, vpatch_size=vpatch_size, vin_chans=vin_chans,
                    num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, drop_path=drop_path,
                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, window_size=window_size,
                    encoder_block=encoder_block, decoder_block=decoder_block, transformer_block=transformer_block, num_res_blocks=num_res_blocks)
    return model


def prostate_inverse_swin_2d_sample_source(simg_size=(3, 1000, 160), spatch_size=(3, 25, 4), sembed_dim=96, sdepth=(2, 2, 6, 2),
                             vimg_size=(400, 160), vpatch_size=(16, 16), vin_chans=1,
                             num_heads=(3, 6, 12, 24), decoder_embed_dim=512, drop_path=0.,
                             mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=5,
                             encoder_block=swin_Encoder, decoder_block=simfwi_Decoder_cnn, transformer_block=Block, num_res_blocks=0, **kwargs):
    sembed_dim = sembed_dim * 2**(len(sdepth)-1)
    model = pde_net(simg_size=simg_size, spatch_size=spatch_size, sembed_dim=sembed_dim, sdepth=sdepth,
                    vimg_size=vimg_size, vpatch_size=vpatch_size, vin_chans=vin_chans,
                    num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, drop_path=drop_path,
                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, window_size=window_size,
                    encoder_block=encoder_block, decoder_block=decoder_block, transformer_block=transformer_block, num_res_blocks=num_res_blocks,
                    sample_source=True)
    return model


def prostate_ae_swin_2d(simg_size=(1, 400, 160), spatch_size=(1, 25, 10), sembed_dim=96, sdepth=(2, 2, 6, 2),
                             vimg_size=(400, 160), vpatch_size=(16, 16), vin_chans=1,
                             num_heads=(3, 6, 12, 24), decoder_embed_dim=512, drop_path=0.,
                             mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=8,
                             encoder_block=swin_Encoder, decoder_block=simfwi_Decoder_cnn, transformer_block=Block, num_res_blocks=0, **kwargs):
    sembed_dim = sembed_dim * 2**(len(sdepth)-1)
    model = pde_net(simg_size=simg_size, spatch_size=spatch_size, sembed_dim=sembed_dim, sdepth=sdepth,
                    vimg_size=vimg_size, vpatch_size=vpatch_size, vin_chans=vin_chans,
                    num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, drop_path=drop_path,
                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, window_size=window_size,
                    encoder_block=encoder_block, decoder_block=decoder_block, transformer_block=transformer_block, num_res_blocks=num_res_blocks)
    return model



if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count, flop_count_str, parameter_count

    # x = torch.rand(3, 3, 45, 1728)
    # y = torch.rand(3, 1, 256, 256)
    # m = finola_y_4norm(simg_size=(45, 1728), sin_chans=3, vimg_size=256, vembed_dim=8192, encoder_block=simfwi_Encoder_input_diff_time,
    #                                         decoder_block=simfwi_Decoder_cnn, sembed_dim=528, num_heads=16, sdepth=3, spatch_size=(9,36),
    #                                         vpatch_size=8, multi_path=8, trans_block=linear_converter)

    x = torch.rand(3, 40, 1000, 160)
    y = torch.rand(3, 1, 400, 160)
    m = prostate_ae_swin_2d()
    params = parameter_count_table(m)
    print(params)

    yy = m(y)

    # for pname, param in m.named_parameters():
    #     print(pname, param.shape)

    # =============================================================================
    # torch.autograd.set_detect_anomaly(True)
    # import timm.optim.optim_factory as optim_factory
    # param_groups = optim_factory.add_weight_decay(m, 0.05)
    # optimizer = torch.optim.AdamW(m.parameters(), lr=0.001, betas=(0.9, 0.95))

    # loss = torch.mean(torch.abs(y - m(x)))
    # print(m.unpatchify(pred).shape)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # t0 = time.time()
    # z = m(x)
    # t1 = time.time()
    # print(t1 - t0)
    #
    # import psutil
    # import os
    #
    # process = psutil.Process(os.getpid())
    #
    # # Parameter memory
    # param_mem = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024 ** 2
    # buffer_mem = sum(b.numel() * b.element_size() for b in m.buffers()) / 1024 ** 2
    #
    # # Before forward pass
    # mem_before = process.memory_info().rss / 1024 ** 2
    #
    # # Forward pass
    # output = m(x)
    #
    # # After forward pass
    # mem_after = process.memory_info().rss / 1024 ** 2
    #
    # # Forward and backward memory
    # forward_mem = mem_after - mem_before
    # backward_mem = forward_mem  # Approximation
    #
    # # Add CUDA overhead (~20%)
    # cuda_overhead = 0.2 * (param_mem + forward_mem + backward_mem)
    # total_cuda_mem = param_mem + forward_mem + backward_mem + cuda_overhead
    #
    # print(f"Parameter memory: {param_mem:.2f} MB")
    # print(f"Buffer memory: {buffer_mem:.2f} MB")
    # print(f"Forward memory: {forward_mem:.2f} MB")
    # print(f"Estimated backward memory: {backward_mem:.2f} MB")
    # print(f"Estimated CUDA memory usage: {total_cuda_mem:.2f} MB")
