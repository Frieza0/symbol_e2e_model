import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意力模块：CBAM
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels)
        )
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.shape
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(b,c)).view(b,c,1,1)
        channel_att = self.sigmoid(avg_out)
        x = x * channel_att
        # 空间注意力
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = self.spatial(torch.cat([avg, max], dim=1))
        x = x * spatial_att
        return x
    
# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 通道最大池化
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out  # 特征加权

# 简单 FPN 模块 
class SimpleFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
    def forward(self, feats_list):
        # 投影到同一维度后拼接
        proj_feats = [proj(feat) for proj, feat in zip(self.projs, feats_list)]
        # 上采样到最大尺寸后拼接
        max_h, max_w = proj_feats[-1].shape[2:]
        proj_feats = [nn.functional.interpolate(feat, size=(max_h, max_w), mode='bilinear') for feat in proj_feats]
        return torch.cat(proj_feats, dim=1)

# 标准 FPN 模块
class StandardFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()  # 横向卷积：将高层特征映射到目标维度
        self.fpn_convs = nn.ModuleList()     # 输出卷积：消除上采样混叠效应
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        # inputs: [f2, f3, f4] 从低层到高层
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        # 自顶向下上采样 + 逐元素相加
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += nn.functional.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], mode='nearest'
            )
        # 输出卷积 + 拼接（或融合）
        outputs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        # 可选：将所有特征上采样到同一尺寸后拼接，提升小引脚特征
        outputs = [nn.functional.interpolate(x, size=outputs[0].shape[2:], mode='nearest') for x in outputs]
        return torch.cat(outputs, dim=1)  # 输出通道数: out_channels * len(in_channels_list)

class RelativePositionEncoding(nn.Module):
    def __init__(self, max_distance, d_model):
        super().__init__()
        self.max_distance = max_distance
        self.embedding = nn.Embedding(2*max_distance +1, d_model)  # 相对距离范围 [-max, max]

    def forward(self, seq_len):
        # 生成相对距离矩阵: [seq_len, seq_len]
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        rel_pos = pos_ids - pos_ids.transpose(0,1)
        # 限制距离范围，避免越界
        rel_pos = torch.clamp(rel_pos, -self.max_distance, self.max_distance)
        rel_pos += self.max_distance  # 映射到 [0, 2*max_distance]
        return self.embedding(rel_pos)