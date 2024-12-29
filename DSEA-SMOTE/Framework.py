import collections
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import argparse
import math
import sys
import random
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn.init as init

# Conv2d_BN: for convolutional layers and batch normalization
class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1, norm_cfg=None):
        super().__init__()

        # Convolutional layer definition
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False
        )

        # Batch Normalization layer definition
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Weight initialization
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, bn_weight_init)
        init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Axial position embedding class
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        # Position embedding layer, initialized to normal distribution
        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        # shape of x should be (B, C_qk, H) or (B, C_qk, W)
        B, C, N = x.shape
        # Perform position embedding interpolation to keep the size matching x
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x


# Sea_Attention module definition
class Sea_Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=2, activation=nn.ReLU, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = nn.Sequential(activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = nn.Sequential(activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        # Fixed parameter ks -> kernel_size
        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, kernel_size=3, stride=1, padding=1, dilation=1, groups=2 * self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, kernel_size=1, norm_cfg=norm_cfg)

    def forward(self, x):  
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = xx.sigmoid() * qkv
        return xx



# DSEA-SMOTE model
class DSEASMOTE(nn.Module):
    def __init__(self):
        super(DSEASMOTE, self).__init__()

        # Encoder part: compressing the input image into the latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Decoder part: decode the latent space representation back to the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Use Sigmoid to output the image (normalized to the range [0, 1])
        )
        
        self.aux_layer = nn.Sequential(nn.Linear(1024,10),nn.Softmax()) # AFC-classfier
        
        # Sea_Attention Enhancement Module
        self.attention = Sea_Attention(dim=512, key_dim=64, num_heads=8)

    def forward(self, x):
        # Encoder: Extracting latent representation of an image
        x = self.encoder(x)

        # Enhance features through Sea_Attention
        x = self.attention(x)

        # Decoder: decode the enhanced features back into an image
        x = self.decoder(x)
        class_output = self.aux_layer(x.view(x.shape[0],-1))
        
        return x,class_output