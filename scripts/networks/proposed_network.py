# デバック確認用
import pdb
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torch.nn.functional as F
# from .DeepLabV3 import DeepLabV3
try:
    from .DeepLabV3 import DeepLabV3
except ImportError:
    from DeepLabV3 import DeepLabV3
    
from .layers import CrossAttention_Mirror, OpticalAttentionModule


class VMD_Network(nn.Module):
    def __init__(self, pretrained_path=None, num_classes=1, all_channel=256, all_dim=26 * 26, T=0.07):  # 473./8=60 416./8=52
        super().__init__()
        self.encoder = DeepLabV3()
        
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            print(f"Load checkpoint:{pretrained_path}")
            self.encoder.load_state_dict(checkpoint['model'])
        
        self.ra_attention_low = Relation_Attention(in_channels=256, out_channels=256)

        self.ra_attention_cross = Relation_Attention(in_channels=256, out_channels=256)

        self.ra_attention_examplar = Relation_Attention(in_channels=256, out_channels=256)
        self.ra_attention_query = Relation_Attention(in_channels=256, out_channels=256)
        self.ra_attention_other = Relation_Attention(in_channels=256, out_channels=256)
        
        # self.cross_attention_module = CrossAttention_Mirror(in_channels=3840, out_channels=3840//2, num_classes=num_classes)
        self.cross_attention_module = OpticalAttentionModule(in_channels=3840, out_channels=3840//2, kernel_size=1)
        
        # reduce dimension
        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.final_pre = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.final_examplar = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

        self.final_query = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

        self.final_other = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
        initialize_weights(self.ra_attention_low, self.ra_attention_cross, self.ra_attention_examplar, self.ra_attention_query, self.ra_attention_other, self.project, self.final_pre, 
        self.final_examplar, self.final_query, self.final_other)

    def forward(self, input1, input2, input3, input1_featmap, input3_featmap, opflow_angle, opflow_magnitude):
        input_size = input1.size()[2:]
        
        low_exemplar, exemplar = self.encoder(input1)
        low_query, query = self.encoder(input2)
        low_other, other = self.encoder(input3)

        #ehnance low level feature
        low_exemplar, low_query = self.ra_attention_low(low_exemplar, low_query)

        # relational attention (cross)
        x1, x2 = self.ra_attention_cross(exemplar, query)
        x1 = F.interpolate(x1, size=low_exemplar.shape[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=low_query.shape[2:], mode='bilinear', align_corners=False)
        x3 = F.interpolate(other, size=low_other.shape[2:], mode='bilinear', align_corners=False)
        fuse_exemplar = torch.cat([x1, self.project(low_exemplar)], dim=1)
        fuse_query = torch.cat([x2, self.project(low_query)], dim=1)
        fuse_other = torch.cat([x3, self.project(low_other)], dim=1)
        exemplar_pre = self.final_pre(fuse_exemplar)
        query_pre = self.final_pre(fuse_query)
        other_pre = self.final_pre(fuse_other)

        # intermidiate prediction
        exemplar_pre = F.interpolate(exemplar_pre, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        query_pre = F.interpolate(query_pre, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        other_pre = F.interpolate(other_pre, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8

        # enhance high level feature
        examplar_pre_small = F.interpolate(exemplar_pre, size=exemplar.shape[2:], mode='bilinear', align_corners=False)
        query_pre_small = F.interpolate(query_pre, size=query.shape[2:], mode='bilinear', align_corners=False)
        other_pre_small = F.interpolate(other_pre, size=other.shape[2:], mode='bilinear', align_corners=False)

        sigmoid_examplar = torch.sigmoid(examplar_pre_small)
        sigmoid_query = torch.sigmoid(query_pre_small)
        sigmoid_other = torch.sigmoid(other_pre_small)

        # 鏡面領域外 (outside) の予測画像を計算
        outside_examplar = torch.ones(sigmoid_examplar.size()).cuda() - sigmoid_examplar
        outside_query =  torch.ones(sigmoid_query.size()).cuda() - sigmoid_query
        outside_other = torch.ones(sigmoid_other.size()).cuda() - sigmoid_other

        outside_query_feat = outside_query * query
        outside_examplar_feat = outside_examplar * exemplar
        outside_other_feat = outside_other * other

        enhanced_examplar, _ = self.ra_attention_examplar(sigmoid_examplar * exemplar, outside_query_feat.transpose(-2, -1))
        enhanced_query, _ = self.ra_attention_query(sigmoid_query * query, outside_examplar_feat.transpose(-2, -1))
        enhanced_other, _ = self.ra_attention_other(sigmoid_other * other, outside_examplar_feat.transpose(-2, -1))

        # proposed
        opflow = torch.stack([opflow_angle, opflow_magnitude], dim=1)
        opflow_based_pred = self.cross_attention_module(input3_featmap, input1_featmap, opflow)
        opflow_based_pred_small = F.interpolate(opflow_based_pred, size=exemplar.shape[2:], mode='bilinear', align_corners=False)
        opflow_based_pred = F.interpolate(opflow_based_pred, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        # pdb.set_trace()
        # original
        final_examplar = self.final_examplar(enhanced_examplar + opflow_based_pred_small)
        final_query = self.final_query(enhanced_query)
        final_other = self.final_other(enhanced_other)

        final_examplar = F.interpolate(final_examplar, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        final_query = F.interpolate(final_query, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        final_other = F.interpolate(final_other, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        
        if self.training:
            return exemplar_pre, query_pre, other_pre, final_examplar, final_query, final_other, opflow_based_pred
        else:
            # return exemplar_pre, query_pre, other_pre, logits
            return final_examplar, final_query, final_other, opflow_based_pred



def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class RAttention(nn.Module):
    '''This part of code is refactored based on https://github.com/Serge-weihao/CCNet-Pure-Pytorch. 
       We would like to thank Serge-weihao and the authors of CCNet for their clear implementation.'''
    def __init__(self,in_dim):
        super(RAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))


    def forward(self, x_exmplar, x_query):
        m_batchsize, _, height, width = x_query.size()
        proj_query = self.query_conv(x_query)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_query_LR = torch.diagonal(proj_query, 0, 2, 3)
        proj_query_RL = torch.diagonal(torch.transpose(proj_query, 2, 3), 0, 2, 3)
        # .contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_key = self.key_conv(x_exmplar)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_key_LR = torch.diagonal(proj_key, 0, 2, 3).permute(0,2,1).contiguous()
        proj_key_RL = torch.diagonal(torch.transpose(proj_key, 2, 3), 0, 2, 3).permute(0,2,1).contiguous()

        proj_value = self.value_conv(x_exmplar)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_value_LR = torch.diagonal(proj_value, 0, 2, 3)
        proj_value_RL = torch.diagonal(torch.transpose(proj_value, 2, 3), 0, 2, 3)

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)

        # energy_LR = torch.bmm(proj_query_LR, proj_key_LR)
        # energy_RL = torch.bmm(proj_query_RL, proj_key_RL)
        energy_LR = torch.bmm(proj_key_LR, proj_query_LR)
        energy_RL = torch.bmm(proj_key_RL, proj_query_RL)


        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        out_LR = self.softmax(torch.bmm(proj_value_LR, energy_LR).unsqueeze(-1))
        out_RL = self.softmax(torch.bmm(proj_value_RL, energy_RL).unsqueeze(-1))

        # print(out_H.size())
        # print(out_LR.size())
        # print(out_RL.size())


        return self.gamma_1*(out_H + out_W + out_LR + out_RL) + x_exmplar, self.gamma_2*(out_H + out_W + out_LR + out_RL) + x_query, 

class Relation_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Relation_Attention, self).__init__()
        inter_channels = in_channels // 4
        self.conv_examplar = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.conv_query = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.ra = RAttention(inter_channels)
        self.conv_examplar_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),nn.ReLU(inplace=False))
        self.conv_query_tail = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),nn.ReLU(inplace=False))

            
    def forward(self, x_exmplar, x_query, recurrence=2):
        # print(x_exmplar.size())
        # print(x_query.size())

        x_exmplar = self.conv_examplar(x_exmplar)
        x_query = self.conv_query(x_query)
        for i in range(recurrence):
            x_exmplar, x_query = self.ra(x_exmplar, x_query)
        x_exmplar = self.conv_examplar_tail(x_exmplar)
        x_query = self.conv_query_tail(x_query)
        return x_exmplar, x_query

        # output = self.conva(x)
        # for i in range(recurrence):
        #     output = self.ra(output)
        # output = self.convb(output)
        
        # return output
        


class CoattentionModel(nn.Module):  # spatial and channel attention module
    def __init__(self, num_classes=1, all_channel=256, all_dim=26 * 26):  # 473./8=60 416./8=52
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate1 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate2 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.globalAvgPool = nn.AvgPool2d(26, stride=1)
        self.fc1 = nn.Linear(in_features=256*2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=256)
        self.fc3 = nn.Linear(in_features=256*2, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, exemplar, query):

        # spatial co-attention
        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim=1)  #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        
        # spacial attention
        input1_mask = self.gate1(torch.cat([input1_att, input2_att], dim=1))
        input2_mask = self.gate2(torch.cat([input1_att, input2_att], dim=1))
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)

        # channel attention
        out_e = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_e = out_e.view(out_e.size(0), -1)
        out_e = self.fc1(out_e)
        out_e = self.relu(out_e)
        out_e = self.fc2(out_e)
        out_e = self.sigmoid(out_e)
        out_e = out_e.view(out_e.size(0), out_e.size(1), 1, 1)
        out_q = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_q = out_q.view(out_q.size(0), -1)
        out_q = self.fc3(out_q)
        out_q = self.relu(out_q)
        out_q = self.fc4(out_q)
        out_q = self.sigmoid(out_q)
        out_q = out_q.view(out_q.size(0), out_q.size(1), 1, 1)

        # apply dual attention masks
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input2_att = out_e * input2_att
        input1_att = out_q * input1_att

        # concate original feature
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = self.bn1(input1_att)
        input2_att = self.bn2(input2_att)
        input1_att = self.prelu(input1_att)
        input2_att = self.prelu(input2_att)

        return input1_att, input2_att  # shape: NxCx

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

if __name__ == '__main__':
    model = VMD_Network().cuda()
    initialize_weights(model)
    exemplar = torch.rand(2, 3, 416, 416)
    query = torch.rand(2, 3, 416, 416)
    other = torch.rand(2, 3, 416, 416)
    exemplar_pre, query_pre, other_pre = model(exemplar, query, other)
    print(exemplar_pre.shape)
    print(query_pre.shape)