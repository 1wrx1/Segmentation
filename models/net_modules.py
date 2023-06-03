import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext50_32x4d, resnet101
from torchvision.models.vgg import vgg16_bn
from torchvision.models.mobilenet import mobilenet_v2


backbone_list = ['vgg16_bn', 'mobilenet_v2', 'resnet18', 'resnet101', 'resnext50_32x4d']


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=False):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = F.relu(x)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        modules = [
            ConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, activation=True),
            ConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True),
        ]
        super().__init__(*modules)


class ConvBottleneck(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        mid_channels = in_channels // 4
        modules = [
            ConvBN(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, activation=True),
            ConvBN(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, activation=True),
            ConvBN(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=True),
        ]
        super().__init__(*modules)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, activation=True)
        self.conv2 = ConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=False)

        self.identical_map = ConvBN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=False)
        self.use_identical = in_channels != out_channels #or stride != 1

    def forward(self, x):
        identity = x
        x = self.conv2(self.conv1(x))

        if self.use_identical:
            identity = self.identical_map(identity)

        x = x + identity
        return F.relu(x)


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = in_channels // 4
        self.conv1 = ConvBN(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, activation=True)
        self.conv2 = ConvBN(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, activation=True)
        self.conv3 = ConvBN(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=False)

        self.identical_map = ConvBN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=False)
        self.use_identical = in_channels != out_channels #or stride != 1

    def forward(self, x):
        identity = x
        x = self.conv3(self.conv2(self.conv1(x)))

        if self.use_identical:
            identity = self.identical_map(identity)

        x = x + identity
        return F.relu(x)


class Backbone(nn.Module):
    def __init__(self, backbone='vgg16_bn', pretrained=False):
        super().__init__()
        assert backbone in backbone_list

        if backbone in ['vgg16_bn', 'mobilenet_v2']:
            self.encoder_list = list(eval(backbone)(pretrained=pretrained).features.children())
            if backbone == 'mobilenet_v2':
                self.nb_filter = [16, 24, 32, 96, 1280]
                self.conv0_0 = nn.Sequential(*self.encoder_list[0:2])  # (None, 64, 112, 112)
                self.conv1_0 = nn.Sequential(*self.encoder_list[2:4])  # (None, 128, 56, 56)
                self.conv2_0 = nn.Sequential(*self.encoder_list[4:7])  # (None, 256, 28, 28)
                self.conv3_0 = nn.Sequential(*self.encoder_list[7:14])  # (None, 512, 14, 14)
                self.conv4_0 = nn.Sequential(*self.encoder_list[14:])  # (None, 512, 7, 7)
            else:
                self.nb_filter = [64, 128, 256, 512, 512]
                self.conv0_0 = nn.Sequential(*self.encoder_list[0:7])  # (None, 64, 112, 112)
                self.conv1_0 = nn.Sequential(*self.encoder_list[7:14])  # (None, 128, 56, 56)
                self.conv2_0 = nn.Sequential(*self.encoder_list[14:24])  # (None, 256, 28, 28)
                self.conv3_0 = nn.Sequential(*self.encoder_list[24:34])  # (None, 512, 14, 14)
                self.conv4_0 = nn.Sequential(*self.encoder_list[34:])  # (None, 512, 7, 7)
            del self.encoder_list
        elif backbone in ['resnet18', 'resnet101', 'resnext50_32x4d']:
            self.encoder_list = list(eval(backbone)(pretrained=pretrained).children())[:-2]
            self.nb_filter = [64, 64, 128, 256, 512] if backbone == 'resnet18' else [64, 256, 512, 1024, 2048]
            self.conv0_0 = nn.Sequential(*self.encoder_list[0:3])  # (None, 64, 112, 112)
            self.conv1_0 = nn.Sequential(*self.encoder_list[3:5])  # (None, 256, 56, 56)
            self.conv2_0 = nn.Sequential(*self.encoder_list[5])  # (None, 512, 28, 28)
            self.conv3_0 = nn.Sequential(*self.encoder_list[6])  # (None, 1024, 14, 14)
            self.conv4_0 = nn.Sequential(*self.encoder_list[7])  # (None, 2048, 7, 7)
            del self.encoder_list

    def forward(self, x):
        x0_0 = self.conv0_0(x)  # (None, 64, 112, 112)
        x1_0 = self.conv1_0(x0_0)  # (None, 256, 56, 56)
        x2_0 = self.conv2_0(x1_0)  # (None, 512, 28, 28)
        x3_0 = self.conv3_0(x2_0)  # (None, 1024, 14, 14)
        x4_0 = self.conv4_0(x3_0)  # (None, 2048, 7, 7)

        out = dict()
        out['x0'] = x0_0
        out['x1'] = x1_0
        out['x2'] = x2_0
        out['x3'] = x3_0
        out['x4'] = x4_0
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, block_type=ResBlock):
        super().__init__()
        assert len(in_channels) == 5
        nb_filter = in_channels
        self.conv3_1 = block_type(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = block_type(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = block_type(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = block_type(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0, x1_0, x2_0, x3_0, x4_0 = x['x0'], x['x1'], x['x2'], x['x3'], x['x4']
        x3_1 = self.conv3_1(torch.cat(
            [x3_0, F.interpolate(x4_0, size=x3_0.shape[-2:], mode='bilinear', align_corners=False)], 1))
        x2_2 = self.conv2_2(torch.cat(
            [x2_0, F.interpolate(x3_1, size=x2_0.shape[-2:], mode='bilinear', align_corners=False)], 1))
        x1_3 = self.conv1_3(torch.cat(
            [x1_0, F.interpolate(x2_2, size=x1_0.shape[-2:], mode='bilinear', align_corners=False)], 1))
        x0_4 = self.conv0_4(torch.cat(
            [x0_0, F.interpolate(x1_3, size=x0_0.shape[-2:], mode='bilinear', align_corners=False)], 1))
        out = self.final(x0_4)

        return out
