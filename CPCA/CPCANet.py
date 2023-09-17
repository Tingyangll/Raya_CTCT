from einops import rearrange
from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import CPCA_config
from thop import profile
from layers import SpatialTransformer as SpatialTransformerself
import utilize


# class Mlp(nn.Module):
#     """ Multilayer perceptron."""
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
# def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
#     result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                      padding=padding, groups=groups)
#     result.add_module('relu', nn.ReLU())
#     return result
#
#
# def fuse_bn(conv_or_fc, bn):
#     std = (bn.running_var + bn.eps).sqrt()
#     t = bn.weight / std
#     t = t.reshape(-1, 1, 1, 1)
#
#     if len(t) == conv_or_fc.weight.size(0):
#         return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
#     else:
#         repeat_times = conv_or_fc.weight.size(0) // len(t)
#         repeated = t.repeat_interleave(repeat_times, 0)
#         return conv_or_fc.weight * repeated, (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(
#             repeat_times, 0)
#
# def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):#  no used
#     result = nn.Sequential()
#     result.add_module('conv', nn.Conv3d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
#                                         bias=False))
#     result.add_module('in', nn.InstanceNorm3d(out_channels))
#     return result


class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,bias=True)
        self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,bias=True)
        self.input_channels = input_channels
        # self.softsign = nn.Softsign()
    def forward(self, inputs):
        x1 = F.adaptive_avg_pool3d(inputs, output_size=(1, 1, 1))
        x1 = self.fc1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool3d(inputs, output_size=(1, 1, 1))
        x2 = self.fc1(x2)
        x2 =  F.leaky_relu(x2, negative_slope=0.2)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1, 1)
        return x

# class ChannelAttention(nn.Module):
#
#     def __init__(self, input_channels, internal_neurons):
#         super(ChannelAttention, self).__init__()
#         self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
#                              bias=True)
#         self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
#                              bias=True)
#         self.input_channels = input_channels
#         self.softsign = nn.Softsign()
#         self.mlp = Mlp(in_features=input_channels, hidden_features=input_channels * 4, drop=0.15)
#
#     def forward(self, inputs):
#         # utilize.show_slice(inputs.cpu().detach().numpy())
#         x1 = F.adaptive_avg_pool3d(inputs, output_size=(1, 1, 1))
#         x1 = self.fc1(x1)
#         x1 = F.leaky_relu(x1, negative_slope=0.2)
#         x1 = self.fc2(x1)
#         b_1,c_1,d_1,h_1,w_1 = x1.shape
#         x1 = x1.view(b_1,c_1*d_1*h_1*w_1)
#         x1 = self.mlp(x1)
#         # utilize.show_slice(x1.cpu().detach().numpy())
#
#         x2 = F.adaptive_max_pool3d(inputs, output_size=(1, 1, 1))
#         # print('x:', x.shape)
#         x2 = self.fc1(x2)
#         x2 = F.leaky_relu(x2, negative_slope=0.2)
#         x2 = self.fc2(x2)
#         b_2, c_2, d_2, h_2, w_2 = x2.shape
#         x2 = x1.view(b_2, c_2 * d_2 * h_2 * w_2)
#         x2 = self.mlp(x2)
#         # utilize.show_slice(x2.cpu().detach().numpy())
#
#         x = x1 + x2
#
#         x = x.view(b_2, c_2, d_2, h_2, w_2)
#         x = self.softsign(x)
#
#         x = x.view(-1, self.input_channels, 1, 1, 1)
#         return x


class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels,channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7_7 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 7, 7), padding=(0, 3, 3), groups=in_channels)
        self.dconv7_7_1 = nn.Conv3d(in_channels, in_channels, kernel_size=(7, 7, 1), padding=(3, 3, 0), groups=in_channels)
        self.dconv7_1_7 = nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 7), padding=(3, 0, 3),groups=in_channels)

        self.dconv1_11_11 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 11, 11), padding=(0, 5, 5 ), groups=in_channels)
        self.dconv11_11_1 = nn.Conv3d(in_channels, in_channels, kernel_size=(11, 11, 1), padding=(5, 5, 0), groups=in_channels)
        self.dconv11_1_11 = nn.Conv3d(in_channels, in_channels, kernel_size=(11, 1, 11), padding=(5, 0, 5),groups=in_channels)

        self.dconv1_21_21 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 21, 21), padding=(0, 10, 10), groups=in_channels)
        self.dconv21_21_1 = nn.Conv3d(in_channels, in_channels, kernel_size=(21, 21, 1), padding=(10, 10, 0), groups=in_channels)
        self.dconv21_1_21 = nn.Conv3d(in_channels, in_channels, kernel_size=(21, 1, 21), padding=(10, 0, 10),groups=in_channels)

        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), padding=0)
        self.act = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, inputs):

        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        #   channel attention
        channel_att_vec = self.ca(inputs)
        out = channel_att_vec * inputs

        # #   spatial attention
        # x_init = self.dconv5_5(inputs)
        # x_1 = self.dconv1_7_7(x_init)
        # x_1 = self.dconv7_7_1(x_1)
        # x_1 = self.dconv7_1_7(x_1)
        #
        # x_2 = self.dconv1_11_11(x_init)
        # x_2 = self.dconv11_11_1(x_2)
        # x_2 = self.dconv11_1_11(x_2)
        #
        # x_3 = self.dconv1_21_21(x_init)
        # x_3 = self.dconv21_21_1(x_3)
        # x_3 = self.dconv21_1_21(x_3)
        #
        # x = x_1 + x_2 + x_3 + x_init
        #
        # # conv(1*1*1)
        # spatial_att = self.conv(x)
        # out = spatial_att * inputs
        # conv(1*1*1)
        out = self.conv(out)
        return out

# class RepBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels,channelAttention_reduce=4):
#         super().__init__()
#
#         self.C = in_channels
#         self.O = out_channels
#
#         assert in_channels == out_channels
#         self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
#         self.dconv5_5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
#         self.dconv1_7_7 = nn.Conv3d(in_channels, in_channels, kernel_size=(7, 7, 7), padding=(3, 3, 3), groups=in_channels)
#         self.dconv1_11_11 = nn.Conv3d(in_channels, in_channels, kernel_size=(11, 11, 11), padding=(5, 5, 5 ), groups=in_channels)
#         self.dconv1_21_21 = nn.Conv3d(in_channels, in_channels, kernel_size=(21, 21, 21), padding=(10, 10, 10), groups=in_channels)
#
#         self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), padding=0)
#         self.act = nn.LeakyReLU(0.2,inplace=True)
#
#     def forward(self, inputs):
#
#         #   Global Perceptron
#         inputs = self.conv(inputs)
#         inputs = self.act(inputs)
#
#         #   channel attention
#         channel_att_vec = self.ca(inputs)
#         # utilize.show_slice(channel_att_vec.cpu().detach().numpy())
#         inputs = channel_att_vec * inputs
#         # utilize.show_slice(inputs.cpu().detach().numpy())
#
#         #   spatial attention
#         x_init = self.dconv5_5(inputs)
#         x_1 = self.dconv1_7_7(x_init)
#         x_2 = self.dconv1_11_11(x_init)
#         x_3 = self.dconv1_21_21(x_init)
#
#         x = x_1 + x_2 + x_3 + x_init
#
#         spatial_att = self.conv(x)
#         out = spatial_att * inputs
#         # conv(1*1*1)
#         out = self.conv(out)
#         return out
#

#   The common FFN Block used in many Transformer and MLP models.

# class FFNBlock(nn.Module):#   no used
#     def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.LeakyReLU(0.2,inplace=True)):
#         super().__init__()
#         out_features = out_channels or in_channels
#         hidden_features = hidden_channels or in_channels
#         self.ffn_fc1 = conv_bn(in_channels, hidden_features, 1, 1, 0)
#         self.ffn_fc2 = conv_bn(hidden_features, out_features, 1, 1, 0)
#         self.act = act_layer()
#
#     def forward(self, x):
#         x = self.ffn_fc1(x)
#         x = self.act(x)
#         x = self.ffn_fc2(x)
#         return x


#   The common FFN Block used in SegneXt models.
class FFNBlock2(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.LeakyReLU(0.2,inplace=True)):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.conv1 = nn.Conv3d(in_channels, hidden_features, kernel_size=(1, 1, 1), padding=0)
        self.conv2 = nn.Conv3d(hidden_channels, out_features, kernel_size=(1, 1, 1), padding=0)
        self.dconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3, 3, 3), padding=(1, 1,1),groups=hidden_features)
        self.act = act_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.dconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
    # def __init__(self, dim, norm_layer=nn.InstanceNorm3d):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
    def forward(self, x, D, H, W):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        # x = F.gelu(x)
        x = self.relu(x)
        # x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.reduction(x)
        return x


class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
    # def __init__(self, dim, norm_layer=nn.InstanceNorm3d):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        # self.up = nn.ConvTranspose3d(dim, dim // 2, 2, 2, 0, 0, bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upconv =nn.Conv3d(in_channels=dim,out_channels=dim//2,kernel_size=1,stride=1,padding=0)
    def forward(self, x, D, H, W):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        # x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3 )
        x = self.up(x)
        x = self.upconv(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., ffn_expand=4, channelAttention_reduce=4):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.In = nn.InstanceNorm3d(dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_block = FFNBlock2(dim, dim * ffn_expand)
        self.repmlp_block = RepBlock(in_channels=dim, out_channels=dim, channelAttention_reduce=channelAttention_reduce)

    def forward(self, x):
        input = x.clone()
        x = self.In(x)
        x = self.repmlp_block(x)
        x = input + self.drop_path(x)
        x2 = self.In(x)
        x2 = self.ffn_block(x2)
        x = x + self.drop_path(x2)

        return x


class Block_up(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1)  # conv
        self.bn = nn.InstanceNorm3d(dim)
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 channelAttention_reduce=4,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 ):
        super().__init__()
        self.depth = depth
        self.dim = dim
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                channelAttention_reduce=channelAttention_reduce
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, D, H, W):

        for blk in self.blocks:
            x = blk(x)
            # utilize.show_slice(x.cpu().detach().numpy())

        if self.downsample is not None:
            x_down = self.downsample(x, D, H, W)
            Wh, Ww, Wd = (H + 1) // 2, (W + 1) // 2,( D + 1) // 2
            return x, H, W, D, x_down, Wd, Wh, Ww
        else:
            return x, H, W, D, x, D, H, W


class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 channelAttention_reduce=4,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 # norm_layer=nn.InstanceNorm3d,
                 upsample=True
                 ):
        super().__init__()
        self.depth = depth
        self.dim = dim

        # build blocks
        self.blocks = nn.ModuleList([
            Block_up(dim=dim)
            for i in range(depth)])

        self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer)
        self.channelConv = nn.Conv3d(in_channels=2*dim,out_channels=dim,kernel_size=1,stride=1,padding=0)

    def forward(self, x, skip, D, H, W):
        # x_up = self.Upsample(x, H, W, D)
        x_up = nn.functional.interpolate(x, skip.shape[2:], mode='trilinear', align_corners=True)
        x_up = self.channelConv(x_up)
        x = x_up + skip
        D, H, W = D * 2, H * 2, W * 2

        for blk in self.blocks:
            x = blk(x)

        return x, D, H, W


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        # x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wd, Wh, Ww)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wd, Wh, Ww = x.size(2), x.size(3),  x.size(4)
            x = x.flatten(2).transpose(1, 2)
            # x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wd, Wh, Ww)
        return x


class project_up(nn.Module):
    def __init__(self, in_dim, out_dim, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        # self.conv1 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upconv = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.upconv(x)
        x = self.activate(x)
        # norm1
        Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        # x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim,Wd, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            # x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wd, Wh, Ww)
        return x

class project_up_raya(nn.Module):
    def __init__(self, in_dim, out_dim, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        # self.conv1 = nn.ConvTranspose3d(in_dim*2, out_dim, kernel_size=2, stride=2)
        self.conv1 =nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upconv = nn.Conv3d(in_dim, out_dim,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv3d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)
        self.channelConv = nn.Conv3d(in_channels=in_dim,out_channels=out_dim,kernel_size=1,stride=1,padding=0)
    def forward(self, x, skips_1_2):
        # x = self.conv1(x)
        # x = self.upconv(x)
        x = nn.functional.interpolate(x, skips_1_2.shape[2:], mode='trilinear', align_corners=True)
        x = self.channelConv(x)
        x = torch.cat((x, skips_1_2), dim=1)
        x = self.activate(x)
        # norm1
        Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        # x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, 2*self.out_dim, Wd, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            # x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wd, Wh, Ww)
        return x

# class PatchEmbed(nn.Module):
#
#     def __init__(self, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
#         super().__init__()
#         self.patch_size = patch_size
#
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#         self.num_block = int(np.log2(patch_size[0]))
#         self.project_block = []
#         self.dim = [int(embed_dim) // (2 ** i) for i in range(self.num_block)]
#         self.dim.append(in_chans)
#         self.dim = self.dim[::-1]  # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim
#
#         for i in range(self.num_block)[:-1]:
#             self.project_block.append(project(self.dim[i], self.dim[i + 1], 2, 1, nn.LeakyReLU(0.2,inplace=True), nn.LayerNorm, False))
#         self.project_block.append(project(self.dim[-2], self.dim[-1], 2, 1, nn.LeakyReLU(0.2,inplace=True), nn.LayerNorm, True))
#         self.project_block = nn.ModuleList(self.project_block)
#
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None
#
#     def forward(self, x):
#         """Forward function."""
#         # padding
#         _, _, H, W, D = x.size()
#
#         out_1_2 = []
#
#         if D % self.patch_size[2] != 0:
#             x = F.pad(x, (0, self.patch_size[2] - D % self.patch_size[2]))
#         if W % self.patch_size[1] != 0:
#             x = F.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
#         if H % self.patch_size[0] != 0:
#             x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
#         for blk in self.project_block:
#              x = blk(x)
#              out_1_2.append(x)
#              # utilize.show_slice(x.cpu().detach().numpy())
#
#         if self.norm is not None:
#             Wh, Ww, Wd = x.size(2), x.size(3), x.size(4)
#             x = x.flatten(2).transpose(1, 2)
#             # x = self.norm(x)
#             x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wd)
#         return x,out_1_2


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block = int(np.log2(patch_size[0]))
        self.project_block = []
        self.dim = [int(embed_dim) // (2 ** i) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim = self.dim[::-1]  # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim

        self.project_block1 = project(self.dim[0], self.dim[1], 2, 1, nn.LeakyReLU(0.2,inplace=True), nn.LayerNorm, False)
        self.project_block2 = project(self.dim[1], self.dim[2], 2, 1, nn.LeakyReLU(0.2,inplace=True), nn.LayerNorm, True)


        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.first_conv = nn.Conv3d(in_channels=in_chans, out_channels=in_chans,kernel_size=3,stride=1,padding=1)
        self.first_conv_1_1_1 = nn.Conv3d(in_channels=in_chans, out_channels=24, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()

        out_1_2 = []

        if D % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - D % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))


        x1 = self.first_conv(x)
        x = self.first_conv_1_1_1(x1)
        out_1_2.append(x)
        x2 = self.project_block1(x1)

        out_1_2.append(x2)
        x = self.project_block2(x2)

        # utilize.show_slice(x.cpu().detach().numpy())

        if self.norm is not None:
            Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            # x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wd, Wh, Ww)


        return x,out_1_2


class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=(160, 160,160),
                 patch_size=(4, 4, 4),
                 in_chans=2,
                 embed_dim=96,
                 depths=(3, 3, 12, 3),
                 channelAttention_reduce=16,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 # norm_layer=nn.InstanceNorm3d,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[1] // 2 ** i_layer
                ),
                depth=depths[i_layer],
                channelAttention_reduce=channelAttention_reduce,
                drop_path=dpr[sum( depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""

        x,out_1_2 = self.patch_embed(x)
        down = []

        Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)

        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, D, x, Wd, Wh, Ww = layer(x, Wd, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = x_out.permute(0, 2, 3, 4, 1)
                # x_out = norm_layer(x_out)
                out = x_out.view(-1, D, H, W,self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                down.append(out)
        return out_1_2,down


class decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=(4, 4),
                 depths=(3, 3, 3),
                 channelAttention_reduce=4,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 # norm_layer=nn.InstanceNorm3d
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[2] // patch_size[2] // 2 ** (len(depths) - i_layer - 1),
                ),

                depth=depths[i_layer],
                channelAttention_reduce=channelAttention_reduce,
                drop_path=dpr[sum(depths[:(len(depths) - i_layer - 1)]):sum(depths[:(len(depths) - i_layer)])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        outs = []
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, D, H, W = layer(x, skips[i], D, H, W)
            outs.append(x)
        return outs


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        # self.num_block = int(np.log2(patch_size[0])) - 2
        self.num_block = int(np.log2(patch_size[0]))
        self.project_block = []
        self.dim_list = [int(dim) // (2 ** i) for i in range(self.num_block + 1)]
        # dim, dim/2, dim/4
        for i in range(self.num_block):
            self.project_block.append(project_up(self.dim_list[i], self.dim_list[i + 1], nn.LeakyReLU(inplace=True), nn.LayerNorm, False))
            # self.project_block.append(project_up(self.dim_list[i], self.dim_list[i + 1], nn.GELU, nn.InstanceNorm3d, False))
        self.project_block = nn.ModuleList(self.project_block)
        # self.up_final = nn.ConvTranspose3d(self.dim_list[-1], num_class, 4, 4)
        self.up_final = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.up_finalconv = nn.Conv3d(in_channels=self.dim_list[-1], out_channels=num_class, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        for blk in self.project_block:
            x = blk(x)
        # x = self.up_final(x)
        # x = self.up_finalconv(x)
        return x


class final_patch_expanding_change(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.project_block = []
        # dim, dim/2, dim/4
        self.project_block0 = project_up_raya(dim, dim//2, nn.LeakyReLU(inplace=True), nn.LayerNorm, False)
        self.project_block1 = project_up_raya(dim//2, dim//4, nn.LeakyReLU(inplace=True), nn.LayerNorm, False)
    def forward(self, x,skips_1_2):
        x1 = self.project_block0(x,skips_1_2[1])
        x =  self.project_block1(x1,skips_1_2[0])
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels = in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=1)
        nn.LeakyReLU(0.2)
        super().__init__(conv3d)

class CPCANet(nn.Module):
    def __init__(self,
                 config,
                 num_input_channels, #1
                 embedding_dim,      #96
                 num_classes,        #[6,12,24,48] or 4
                 deep_supervision,   #False
                 conv_op=nn.Conv3d):
        super(CPCANet, self).__init__()

        # Don't uncomment conv_op
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.do_ds = deep_supervision
        self.embed_dim = embedding_dim
        self.depths = config.hyper_parameter.blocks_num #[3,3,12,3]
        self.crop_size = config.hyper_parameter.crop_size
        self.patch_size = [config.hyper_parameter.convolution_stem_down, config.hyper_parameter.convolution_stem_down, config.hyper_parameter.convolution_stem_down]#[4,4,4]
        self.channelAttention_reduce = config.hyper_parameter.channelAttention_reduce #16
        # if window size of the encoder is [7,7,14,7], then decoder's is [14,7,7]. In short, reverse the list and start from the index of 1
        self.model_down = encoder(
            pretrain_img_size=self.crop_size,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            depths=self.depths,
            in_chans=self.num_input_channels,
            channelAttention_reduce=self.channelAttention_reduce

        )

        self.decoder = decoder(
            pretrain_img_size=self.crop_size,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            depths=[2, 2, 1],
            channelAttention_reduce=self.channelAttention_reduce
        )

        self.final = []
        for i in range(len(self.depths) - 1):
            self.final.append(final_patch_expanding(self.embed_dim * 2 ** i, self.num_classes, patch_size=self.patch_size))
        self.final = nn.ModuleList(self.final)
        self.final_raya = final_patch_expanding_change(self.embed_dim, self.num_classes, patch_size=self.patch_size)
        self.head = RegistrationHead(in_channels=24, out_channels=3, kernel_size=3)
        self.STN = SpatialTransformerself()
    def forward(self, x,training = True):
        # b,c,d,h,w
        source = x[:, 0:1, :, :]
        # utilize.show_slice(source.cpu().detach().numpy())
        seg_outputs = []
        skips_1_2,skips = self.model_down(x)
        # utilize.show_slice(skips_1_2[0].cpu().detach().numpy())
        # utilize.show_slice(skips_1_2[1].cpu().detach().numpy())
        # utilize.show_slice(skips[0].cpu().detach().numpy())
        # utilize.show_slice(skips[1].cpu().detach().numpy())
        # utilize.show_slice(skips[2].cpu().detach().numpy())
        # utilize.show_slice(skips[3].cpu().detach().numpy())
        neck = skips[-1]
        out = self.decoder(neck, skips)
        # utilize.show_slice(out[0].cpu().detach().numpy())
        # utilize.show_slice(out[1].cpu().detach().numpy())
        # utilize.show_slice(out[2].cpu().detach().numpy())

        # for i in range(len(out)):
        #     seg_outputs.append(self.final[-(i + 1)](out[i]))

        #add two skips
        seg_outputs =self.final_raya(out[-1],skips_1_2)
        seg_outputs = self.head(seg_outputs)
        # seg_outputs[-1] = self.head(seg_outputs[-1])

        if training:
            out = self.STN(source, seg_outputs)
            # out = self.STN(source, seg_outputs[-1])
            # utilize.show_slice(out.cpu().detach().numpy())
            # utilize.show_slice(seg_outputs[-1].cpu().detach().numpy())
            return out, seg_outputs

        else:
            return seg_outputs
            # return seg_outputs[-1]


# if __name__ == '__main__':
#     with torch.no_grad():
#         import os
#
#         os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#         cuda0 = torch.device('cuda:0')
#
#         x = torch.rand((1, 2, 160, 160,160), device=cuda0)  #b,c,d,h,w
#         config = CPCA_config['ACDC_224']
#         model = CPCANet(config,
#                         2,
#                         96,
#                         [6, 12, 24, 48],
#                         # 4,
#                         False,
#                         conv_op=nn.Conv3d)
#         model.cuda()
#         y = model(x)
#         print(y.shape)
#         hereflops, params = profile(model, inputs=(x,))
#         print("hereflops:", hereflops)
#         print("params:", params)





