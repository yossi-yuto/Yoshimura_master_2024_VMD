import torch
import torch.nn as nn


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


class CrossAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int = 1, kernel_size: int = 1):
        super(CrossAttention, self).__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.ca = CrossAttentionBlock(out_channels, out_channels, kernel_size)
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.sa = CrossAttentionBlock(out_channels, out_channels, kernel_size)
        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_pred = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = self.conv1x1(x), self.conv1x1(y)
        x = self.ca(x, y)
        x = self.conv_bn_relu1(x)
        x = self.sa(x, y)
        x = self.conv_bn_relu2(x)
        x = self.conv_pred(x)
        return x