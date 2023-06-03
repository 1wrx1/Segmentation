from .net_modules import *

block_dict = {'vgg16_bn': ConvBlock,
              'mobilenet_v2': ConvBottleneck,
              'resnet18': ResBlock,
              'resnet101': ResBottleneck,
              'resnext50_32x4d': ResBottleneck}


class AttentionGate(nn.Module):
    """
    Attention U-Net中使用的注意力门结构
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * self.upsample(psi)


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation (arXiv:1505.04597)
    """

    def __init__(self, num_classes, backbone='vgg16_bn', pretrained=False):
        super().__init__()
        assert backbone in backbone_list, 'Backbone type not supported!'
        self.backbone = Backbone(backbone=backbone, pretrained=pretrained)
        nb_filter = self.backbone.nb_filter

        conv_type = block_dict[backbone]

        self.conv3_1 = conv_type(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = conv_type(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = conv_type(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = conv_type(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x0_0, x1_0, x2_0, x3_0, x4_0 = x['x0'], x['x1'], x['x2'], x['x3'], x['x4']

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        output = self.final(self.up(x0_4))

        return output


class NestedUNet(nn.Module):
    """
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation (arXiv:1807.10165)
    """

    def __init__(self, num_classes, backbone='vgg16_bn', pretrained=False, deep_supervision=True):
        super().__init__()
        assert backbone in backbone_list, 'Backbone type not supported!'
        self.backbone = Backbone(backbone=backbone, pretrained=pretrained)
        nb_filter = self.backbone.nb_filter

        conv_type = block_dict[backbone]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv0_1 = conv_type(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = conv_type(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = conv_type(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = conv_type(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = conv_type(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = conv_type(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = conv_type(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = conv_type(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = conv_type(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = conv_type(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x0_0, x1_0, x2_0, x3_0, x4_0 = x['x0'], x['x1'], x['x2'], x['x3'], x['x4']

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(self.up(x0_1))
            output2 = self.final2(self.up(x0_2))
            output3 = self.final3(self.up(x0_3))
            output4 = self.final4(self.up(x0_4))
            output = (output1 + output2 + output3 + output4) / 4
        else:
            output = self.final(self.up(x0_4))

        return output


class AttentionUNet(nn.Module):
    """
    Attention U-Net: Learning Where to Look for the Pancreas (arXiv:1804.03999)
    """

    def __init__(self, num_classes, backbone='vgg16_bn', pretrained=False):
        super().__init__()
        assert backbone in backbone_list, 'Backbone type not supported!'
        self.backbone = Backbone(backbone=backbone, pretrained=pretrained)
        nb_filter = self.backbone.nb_filter

        conv_type = block_dict[backbone]

        self.conv3_1 = conv_type(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = conv_type(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = conv_type(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = conv_type(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.gate4_3 = AttentionGate(nb_filter[4], nb_filter[3], nb_filter[3])
        self.gate3_2 = AttentionGate(nb_filter[3], nb_filter[2], nb_filter[2])
        self.gate2_1 = AttentionGate(nb_filter[2], nb_filter[1], nb_filter[1])
        self.gate1_0 = AttentionGate(nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x0_0, x1_0, x2_0, x3_0, x4_0 = x['x0'], x['x1'], x['x2'], x['x3'], x['x4']

        x3_1 = self.conv3_1(torch.cat([self.gate4_3(x4_0, x3_0), self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([self.gate3_2(x3_1, x2_0), self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([self.gate2_1(x2_2, x1_0), self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([self.gate1_0(x1_3, x0_0), self.up(x1_3)], 1))

        output = self.final(self.up(x0_4))
        return output


class DLinkNet(nn.Module):
    """
    D_LinkNet:  LinkNet with Pretrained Encoder and Dilated Convolution for High
                Resolution Satellite Imagery Road Extraction (arXiv: )
    """

    def __init__(self, num_classes, backbone='resnet101', pretrained=False):
        super().__init__()
        assert backbone in backbone_list, 'Backbone type not supported!'
        self.backbone = Backbone(backbone=backbone, pretrained=pretrained)
        nb_filter = self.backbone.nb_filter

        conv_type = block_dict[backbone]

        self.conv3_1 = conv_type(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = conv_type(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = conv_type(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = conv_type(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.d_conv1 = nn.Conv2d(nb_filter[4], nb_filter[4], kernel_size=3, stride=1, padding=1, dilation=1)
        self.d_conv2 = nn.Conv2d(nb_filter[4], nb_filter[4], kernel_size=3, stride=1, padding=2, dilation=2)
        self.d_conv3 = nn.Conv2d(nb_filter[4], nb_filter[4], kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv_mid=nn.Conv2d(4*nb_filter[4],nb_filter[4],kernel_size=3,stride=1,padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.bn=nn.BatchNorm2d(nb_filter[4])


    def forward(self, x):
        x = self.backbone(x)
        x0_0, x1_0, x2_0, x3_0, x4_0 = x['x0'], x['x1'], x['x2'], x['x3'], x['x4']

        x_m1 = x4_0
        x_m2 = self.d_conv1(x_m1)
        x_m3 = self.d_conv2(self.d_conv1(x_m1))
        x_m4 = self.d_conv3(self.d_conv2(self.d_conv1(x_m1)))
        x4_0 = torch.cat([x_m1,x_m2,x_m3,x_m4],1)
        x4_0= self.conv_mid(x4_0)
        x4_0=self.bn(x4_0)
        x4_0=F.relu(x4_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        output = self.final(self.up(x0_4))

        return output
