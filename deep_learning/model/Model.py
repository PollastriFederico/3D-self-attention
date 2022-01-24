from deep_learning.model.transformer import ViT

import torch
import torch.nn as nn
from deep_learning.model.ResNet3D import ResNet3D, ResNet2Puls1D, ResNetM3D


class MultiScaleAttentionBlock(nn.Module):

    def __init__(self, planes=512, reduction_factor=8):
        super(MultiScaleAttentionBlock, self).__init__()
        self.out_planes = planes // reduction_factor
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(3, 3, 3), stride=1,
                      padding=(1, 1, 1), dilation=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(3, 3, 3), stride=1,
                      padding=(4, 4, 4), dilation=(4, 4, 4), bias=False),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(3, 3, 3), stride=1,
                      padding=(8, 8, 8), dilation=(8, 8, 8), bias=False),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(3, 3, 3), stride=1,
                      padding=(12, 12, 12), dilation=(12, 12, 12), bias=False),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(in_channels=planes, out_channels=self.out_planes, kernel_size=(1, 1, 1), bias=False),
        )
        self.after_pool = nn.Sequential(
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True)
        )

        self.transformer = ViT(
            dim=(14, 18, 18),
            depth=6,
            heads=16,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        out_0 = self.conv0(x)
        out_1 = self.conv1(x)
        out_2 = self.conv2(x)
        out_3 = self.conv3(x)

        out_4 = self.pool(x)
        out_4 = nn.functional.interpolate(out_4, out_1.shape[-3:])
        out_4 = self.after_pool(out_4)

        final = torch.cat((out_0, out_1, out_2, out_3, out_4), dim=1)
        final = self.transformer(final)
        return final


class Model(nn.Module):
    def __init__(self, backbone='ResNet3D', num_classes=1, reduction_factor=8):
        super(Model, self).__init__()
        self.bottle_planes = (512 // reduction_factor) * 4
        if backbone == 'ResNet3D':
            self.resnet = ResNet3D()
        elif backbone == 'ResNet2Plus1D':
            self.resnet = ResNet2Puls1D()
        elif backbone == 'ResNetM3D':
            self.resnet = ResNetM3D()
        self.resnet.set_output_stride()
        self.mslayer = MultiScaleAttentionBlock()

        self.bottle_planes = (512 // reduction_factor) * 5
        mid_planes = self.bottle_planes
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels=self.bottle_planes, out_channels=mid_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_planes, out_channels=num_classes, kernel_size=(1, 1, 1), stride=1, bias=False),
        )

    def forward(self, x):
        out, low_level = self.resnet(x)
        out = self.mslayer(out)
        out = nn.functional.interpolate(out, scale_factor=(4, 4, 4), mode='trilinear')
        out = self.decoder(out)
        out = nn.functional.interpolate(out, scale_factor=(1, 2, 2), mode='trilinear')
        return out


if __name__ == '__main__':
    batch_size = 2
    x = torch.rand(size=(batch_size, 3, 56, 144, 144), device='cuda')
    y = torch.zeros(size=(batch_size, 1, 56, 144, 144), device='cuda')
    model = Model(reduction_factor=8).to('cuda')
    out = model(x)
    loss = torch.nn.BCEWithLogitsLoss()(out, y)
    loss.backward()
    pass
