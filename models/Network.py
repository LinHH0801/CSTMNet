import torch
import torch.nn as nn
from torch.nn import functional as F

from mamba_ssm import Mamba

from thop import profile
from models.pvtv2 import ourpvt_v2_b2,ourpvt_v2_b1,ourpvt_v2_b0,ourpvt_v2_b3,ourpvt_v2_b4

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()

        self.chanel_in = in_dim
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,y):
        indentity = x
        x = x.flatten(2).permute(0, 2, 1)

        query = self.query(y).permute(0, 2, 1).mean(dim=2)
        key = self.key(x).permute(0, 2, 1).mean(dim=2)
        energy = query*key
        attention = self.softmax(energy).unsqueeze(-1).unsqueeze(-1)

        out = indentity+ attention * indentity

        return out



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SptialCNN(nn.Module):
    def __init__(self, in_channels=3,pretrained_path=r'E:\Lp_8\WUSU\models\pvt_v2_b1.pth'):
        super(SptialCNN, self).__init__()
        self.backbone = ourpvt_v2_b1()  # [64, 128, 320, 512]
        save_model = torch.load(pretrained_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.Translayer0 = BasicConv2d(64, 128, 3, padding=1)
        self.Translayer1 = BasicConv2d(128, 128, 3, padding=1)
        self.Translayer2 = BasicConv2d(320, 128, 3, padding=1)
        self.Translayer3 = BasicConv2d(512, 128, 3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv = self._make_layer(ResBlock, 128 * 4, 512, 1, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):

        x1 = self.backbone(x1)
        x1_1 = self.Translayer0(x1[0])
        x1_2 = self.up1(self.Translayer1(x1[1]))
        x1_3 = self.up2(self.Translayer2(x1[2]))
        x1_4 = self.up3(self.Translayer3(x1[3]))
        x1_4 = self.conv(torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1))

        x2 = self.backbone(x2)
        x2_1 = self.Translayer0(x2[0])
        x2_2 = self.up1(self.Translayer1(x2[1]))
        x2_3 = self.up2(self.Translayer2(x2[2]))
        x2_4 = self.up3(self.Translayer3(x2[3]))
        x2_4 = self.conv(torch.cat([x2_1, x2_2, x2_3, x2_4], dim=1))

        x3 = self.backbone(x3)
        x3_1 = self.Translayer0(x3[0])
        x3_2 = self.up1(self.Translayer1(x3[1]))
        x3_3 = self.up2(self.Translayer2(x3[2]))
        x3_4 = self.up3(self.Translayer3(x3[3]))
        x3_4 = self.conv(torch.cat([x3_1, x3_2, x3_3, x3_4], dim=1))

        return x1_4, x2_4, x3_4


class MambaBlock_Channel(nn.Module):
    def __init__(self, hidden_dim, d_state=128, d_conv=4, expand=2, channel_first=True,downsample_ratio=2):
        super().__init__()
        self.channel_first = channel_first
        self.mamba1 = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba2 = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.downsample_ratio = downsample_ratio
        self.linear1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.LN = nn.LayerNorm(512)
        self.norm = nn.LayerNorm(hidden_dim * downsample_ratio ** 2)
        self.reduction = nn.Linear(hidden_dim * downsample_ratio ** 2, hidden_dim)
        self.GAP = nn.AdaptiveAvgPool2d(8)
        self.MAP = nn.AdaptiveMaxPool2d(8)
        self.pool = nn.MaxPool1d(2)

        self.fc1G = nn.Conv2d(hidden_dim, hidden_dim // 4, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, x1,x2):

        xG = self.GAP(torch.cat([x1,x2],dim=1))
        xM = self.MAP(torch.cat([x1,x2],dim=1))

        xG = xG.flatten(2).permute(0, 2, 1)
        xM = xM.flatten(2).permute(0, 2, 1)

        xG = self.linear1(self.mamba1(xG))
        xM = self.linear2(self.mamba2(xM))

        x = self.sigmoid(xG+xM)
        return x

class MambaBlock_Temporal(nn.Module):
    def __init__(self, hidden_dim, d_state=128, d_conv=4, expand=2, channel_first=True,downsample_ratio=2):
        super().__init__()
        self.channel_first = channel_first
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.downsample_ratio = downsample_ratio
        self.linear1 = nn.Linear(hidden_dim, hidden_dim//4)
        self.linear2 = nn.Linear(hidden_dim//4, hidden_dim)
        self.LN = nn.LayerNorm(512)
        self.norm = nn.LayerNorm(hidden_dim * downsample_ratio ** 2)
        self.reduction = nn.Linear(hidden_dim * downsample_ratio ** 2, hidden_dim)
        self.GAP = nn.AdaptiveAvgPool2d(8)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x1,x2):

        x1 = self.GAP(x1)
        x2 = self.GAP(x2)

        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)

        x = torch.cat([x1,x2],dim=1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)

        return x


class MambaBlock_Spatial(nn.Module):
    def __init__(self, hidden_dim, d_state=128, d_conv=4, expand=2, channel_first=True, downsample_ratio=2):
        super().__init__()
        self.channel_first = channel_first
        self.mamba = Mamba(
            d_model=hidden_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.downsample_ratio = downsample_ratio
        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.linear2 = nn.Linear(hidden_dim // 4, hidden_dim//2)
        self.LN = nn.LayerNorm(512)
        self.norm = nn.LayerNorm(hidden_dim * downsample_ratio ** 2)
        self.reduction = nn.Linear(hidden_dim * downsample_ratio ** 2, hidden_dim)
        self.GAP = nn.AdaptiveAvgPool2d(8)


    def forward(self, x1, x2):

        x1 = self.GAP(x1)
        x2 = self.GAP(x2)

        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)

        x = torch.cat([x1, x2], dim=2)
        x = self.linear1(x)
        x = self.mamba(x)
        x = self.linear2(x)

        return x


class CSTMNet(nn.Module):
    def __init__(self, in_channels=3,num_classes=2):
        super(CSTMNet, self).__init__()
        self.Sptial = SptialCNN(in_channels)
        self.conv_cat1 = nn.Linear(512*3, 512)
        self.conv_cat2 = nn.Conv2d(512*2, 512, kernel_size=3, padding=1)
        self.SSM_Channel = MambaBlock_Channel(hidden_dim=512*2, d_state=32)
        self.SSM_Temporal = MambaBlock_Temporal(hidden_dim=512, d_state=32)
        self.SSM_Spatial = MambaBlock_Spatial(hidden_dim=512*2, d_state=32)
        self.SA = SA(512)
        self.head1 = self._make_layer(ResBlock, 512, 64, 1, stride=1)
        self.classifier1 = nn.Conv2d(64, num_classes+2, kernel_size=3, padding=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2,x3):
        x_size = x1.size()

        x1,x2,x3 = self.Sptial(x1, x2,x3)

        diff_12 = (torch.abs(x1-x2))
        diff_23 = (torch.abs(x2-x3))
        mamba_chaannel = self.SSM_Channel(diff_12, diff_23)
        mamba_temporal = self.SSM_Temporal(diff_12,diff_23)
        mamba_spatial= self.SSM_Spatial(diff_12, diff_23)

        mamba = self.conv_cat1(torch.cat([mamba_chaannel,mamba_temporal,mamba_spatial],dim=2))

        y = self.conv_cat2(torch.cat([diff_12,diff_23],dim=1))

        mamba_feature = self.SA(y, mamba)

        change_moments = self.classifier1(self.head1(mamba_feature))

        return F.interpolate(change_moments, x_size[2:], mode='bilinear')


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    model =  CSTMNet(num_classes=2).cuda()
    all_change = model(x1,x2,x1)
    print(all_change.shape)
    flops, params = profile(model, inputs=(x1, x1,x1))[0], profile(model, inputs=(x1, x1,x1))[1]
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
