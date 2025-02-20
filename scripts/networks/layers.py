import pdb
import math

import torch
import torch.nn as nn


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super(CrossAttentionBlock, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.conv_q(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.conv_k(y).view(B, -1, H * W)
        v = self.conv_v(y).view(B, -1, H * W)
        
        energy = torch.bmm(q, k)
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(B, -1, H, W)
        out = self.gamma * out + x
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super(SelfAttentionBlock, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.conv_q(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.conv_k(x).view(B, -1, H * W)
        v = self.conv_v(x).view(B, -1, H * W)
        
        energy = torch.bmm(q, k)
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(B, -1, H, W)
        out = self.gamma * out + x
        return out


class CrossAttention_Mirror(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int = 1, kernel_size: int = 1):
        super(CrossAttention_Mirror, self).__init__()
        
        self.in_channels = in_channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.ca = CrossAttentionBlock(out_channels, out_channels, kernel_size)
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.sa = SelfAttentionBlock(out_channels+2, out_channels+2, kernel_size)
        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(out_channels+2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_pred = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x, y = self.conv1x1(x), self.conv1x1(y)
        _ = self.ca(x, y)
        cross_out = self.conv_bn_relu1(_)
        concat_ = torch.cat([cross_out, flow], dim=1)
        _ = self.sa(concat_)
        _ = self.conv_bn_relu2(_)
        out = self.conv_pred(_)
        return out


class OpticalAttentionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int = 1, kernel_size: int = 1):
        super(OpticalAttentionModule, self).__init__()
        
        self.in_channels = in_channels
        self.conv_channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels+2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pe = positionalencoding2d(out_channels, 52, 52).cuda()
        self.pe.requires_grad = False
        
        self.ca = CrossAttentionBlock(out_channels, out_channels, kernel_size)
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pred = nn.Conv2d(out_channels, num_classes, kernel_size=3, padding=1)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        
        query_frame = torch.cat([query, flow], dim=1)
        key_frame = torch.cat([key, flow], dim=1)
        
        query_frame, key_frame = self.conv_channel_reduction(query_frame), self.conv_channel_reduction(key_frame)
        
        query_frame_with_pos, key_frame_with_pos = query_frame + self.pe[None], key_frame + self.pe[None]
        # cross attention
        _ = self.ca(query_frame_with_pos, key_frame_with_pos)
        # feed forward
        _ = self.conv_bn_relu1(_)
        out = self.pred(_)
        
        return out