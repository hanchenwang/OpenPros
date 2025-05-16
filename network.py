#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

# Replace the key names in the checkpoint in which legacy network building blocks are used 
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)
 

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    
# (B, 40, 1000, 161) -> (B, 1, 401, 161)
class FCN4_Deep(nn.Module):
    def __init__(self, enc_ch=[6, 6, 6, 7, 7, 8, 8, 9], enc_side=[1, 0, 0, 0, 0, 0], 
                 bottle_conv=(8, 6), bottle_deconv=(7, 3), dec_ch=[9, 8, 7, 6, 5, 4, 3],
                 crop=[-15, -16, -23, -24], **kwargs):
        super(FCN4_Deep, self).__init__()
        assert len(enc_ch) == len(enc_side) + 2 # 2 for first and last layer of encoder
        enc_ch = [2**c for c in enc_ch]
        dec_ch = [2**c for c in dec_ch]
        self.crop = crop
        bottle_conv = tuple(bottle_conv)
        bottle_deconv = tuple(bottle_deconv)

        layers = [ConvBlock(40, enc_ch[0], kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))]
        for i in range(1, len(enc_ch) - 1):
            if enc_side[i - 1]:
                layers.append(ConvBlock(enc_ch[i-1], enc_ch[i], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)))
                layers.append(ConvBlock(enc_ch[i], enc_ch[i], kernel_size=(3, 1), padding=(1, 0)))
            else:
                layers.append(ConvBlock(enc_ch[i-1], enc_ch[i], stride=2))
                layers.append(ConvBlock(enc_ch[i], enc_ch[i]))
        layers.append(ConvBlock(enc_ch[-2], enc_ch[-1], kernel_size=bottle_conv, padding=0))
        self.encoder = nn.Sequential(*layers)

        layers = []
        for i in range(len(dec_ch)):
            if i == 0:
                layers.append(DeconvBlock(enc_ch[-1], dec_ch[i], kernel_size=bottle_deconv))
            else:
                layers.append(DeconvBlock(dec_ch[i-1], dec_ch[i], kernel_size=4, stride=2, padding=1))
            layers.append(ConvBlock(dec_ch[i], dec_ch[i]))
        self.decoder = nn.Sequential(*layers)
        self.output = ConvBlock_Tanh(dec_ch[-1], 1)

    def forward(self, x):
        # Encoder Part
        # 500, 250, 125,  63,  32,  16,   8
        # 161, 161,  81,  41,  21,  11,   6
        #  64,  64,  64, 128, 128, 256, 256
        # Decoder Part
        # 7 * 2**6 = 448, 3 * 2**6 = 192
        # crop = [-15, -16, -23, -24]
        # output: 401, 161
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.pad(x, self.crop, mode="constant", value=0)
        x = self.output(x)
        return x


# (B, 40, 1000, 161) -> (B, 1, 401, 161)
class FCN4_Deep_P(nn.Module):
    def __init__(self, dim0=8, dim1=16, dim2=32, dim3=64, dim4=128, dim5=256, dim6=512, **kwargs):
        super(FCN4_Deep_P, self).__init__()
        self.convblock1 = ConvBlock(40, dim3, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim3, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock5_2 = ConvBlock(dim4, dim4)
        self.convblock6_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock6_2 = ConvBlock(dim5, dim5)
        self.convblock7_1 = ConvBlock(dim5, dim5, stride=2)
        self.convblock7_2 = ConvBlock(dim5, dim5)
        self.convblock8 = ConvBlock(dim5, dim6, kernel_size=(8, 6), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim6, dim6, kernel_size=(7, 3))
        self.deconv1_2 = ConvBlock(dim6, dim6)
        self.deconv2_1 = DeconvBlock(dim6, dim5, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim5, dim5)
        self.deconv3_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim4, dim4)
        self.deconv4_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim3, dim3)
        self.deconv5_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim2, dim2)
        self.deconv6_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv6_2 = ConvBlock(dim1, dim1)
        self.deconv7_1 = DeconvBlock(dim1, dim0, kernel_size=4, stride=2, padding=1)
        self.deconv7_2 = ConvBlock(dim0, dim0)
        self.deconv8 = ConvBlock_Tanh(dim0, 1)
        
    def forward(self,x):
        # Encoder Part
        # 500, 250, 125,  63,  32,  16,   8
        # 161, 161,  81,  41,  21,  11,   6
        #  64,  64,  64, 128, 128, 256, 256
        # 
        x = self.convblock1(x) # (None, 64, 500, 161)
        x = self.convblock2_1(x) # (None, 64, 250, 161)
        x = self.convblock2_2(x) # (None, 64, 250, 161)
        x = self.convblock3_1(x) # (None, 64, 125, 81)
        x = self.convblock3_2(x) # (None, 64, 125, 81)
        x = self.convblock4_1(x) # (None, 128, 63, 41) 
        x = self.convblock4_2(x) # (None, 128, 63, 41)
        x = self.convblock5_1(x) # (None, 128, 32, 21) 
        x = self.convblock5_2(x) # (None, 128, 32, 21)
        x = self.convblock6_1(x) # (None, 256, 16, 11) 
        x = self.convblock6_2(x) # (None, 256, 16, 11)
        x = self.convblock7_1(x) # (None, 256, 8, 6) 
        x = self.convblock7_2(x) # (None, 256, 8, 6)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 7, 3)
        x = self.deconv1_2(x) # (None, 512, 7, 3)
        x = self.deconv2_1(x) # (None, 256, 14, 6) 
        x = self.deconv2_2(x) # (None, 256, 14, 6)
        x = self.deconv3_1(x) # (None, 128, 28, 12) 
        x = self.deconv3_2(x) # (None, 128, 28, 12)
        x = self.deconv4_1(x) # (None, 64, 56, 24) 
        x = self.deconv4_2(x) # (None, 64, 56, 24)
        x = self.deconv5_1(x) # (None, 32, 112, 48)
        x = self.deconv5_2(x) # (None, 32, 112, 48)
        x = self.deconv6_1(x) # (None, 16, 224, 96)
        x = self.deconv6_2(x) # (None, 16, 224, 96)
        x = self.deconv7_1(x) # (None, 8, 448, 192)
        x = self.deconv7_2(x) # (None, 8, 448, 192)
        x = F.pad(x, [-15, -16, -23, -24], mode="constant", value=0) # (None, 8, 401, 161)
        x = self.deconv8(x) # (None, 1, 401, 161)
        return x

model_dict = {
    'FCN4_Deep': FCN4_Deep,
    'FCN4_Deep_P': FCN4_Deep_P,
}

if __name__ == '__main__':
    model = FCN4_Deep() # 20447515
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: %d' % total_params)
    x= torch.rand((2, 40, 1000, 161))
    y = model(x)
    print(y.shape)

