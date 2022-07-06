"""
Single-modal version of the MSFCF-Net: S-MSFCF-Net

S-MSFCF-Net removes the backbone network for mask input (BN-M) and 
dual-modal feature fusion modules (DMFF) while the multi-scale feature fusion 
modules (MSFF) will fuse different scales of feature maps from the backbone
network for voltage input (BN-V) like MSFCF-Net does.

@author: LIU Zhe
"""
import torch
import torch.nn as nn
import math


# =====================================================================================
#                                  Model Components
# =====================================================================================

# --------------------------------  Basic Modules  ------------------------------------


class FCLeaky(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.leaky_relu(self.fc(x))


class FCBNLeaky(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.fc(x)))


class ConvLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.leaky_relu(self.conv(x))


class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))


class BNLeakyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(self.leaky_relu(self.bn(x)))


class TconvLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, output_padding=output_padding, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.leaky_relu(self.tconv(x))


class TconvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, output_padding=output_padding, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.tconv(x)))


# --------------------------------  Residual Modules  --------------------------------


class ResUnit(nn.Module):
    # Residual V2 unit.
    # Can choose whether adding Batch Normalization !
    # Input size equals to output size.
    def __init__(self, in_channels, add_bn):
        super().__init__()
        self.add_bn = add_bn
        if add_bn is False:
            self.layer1 = ConvLeaky(in_channels, math.ceil(in_channels / 2), kernel_size=1)
            self.layer2 = ConvLeaky(math.ceil(in_channels / 2), math.ceil(in_channels / 2), kernel_size=3, padding=1)
            self.layer3 = nn.Conv2d(math.ceil(in_channels / 2), in_channels,  kernel_size=1)
            self.layer4 = nn.LeakyReLU(negative_slope=0.01)
        elif add_bn is True:
            self.layer1 = BNLeakyConv(in_channels, math.ceil(in_channels / 2), kernel_size=1)
            self.layer2 = BNLeakyConv(math.ceil(in_channels / 2), math.ceil(in_channels / 2), kernel_size=3, padding=1)
            self.layer3 = BNLeakyConv(math.ceil(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        if self.add_bn is False:
            return self.layer4(x + self.layer3(self.layer2(self.layer1(x))))
        elif self.add_bn is True:
            return x + self.layer3(self.layer2(self.layer1(x)))


class ResBlock(nn.Module):
    # Size UNCHANGEABLE. The height, width and depth are both unchangeable.
    # Can choose whether adding Batch Normalization !
    def __init__(self, in_channels, num_units, add_bn):
        super().__init__()
        self.num_units = num_units
        self.res_units = nn.ModuleList([])
        if add_bn is False:
            for i in range(num_units):
                self.res_units.append(ResUnit(in_channels, add_bn=False))
        elif add_bn is True:
            for i in range(num_units):
                self.res_units.append(ResUnit(in_channels, add_bn=True))

    def forward(self, x):
        for i in range(self.num_units):
            x = self.res_units[i](x)
        return x


class ResBlockCC(nn.Module):
    # Channel CHANGEABLE. The height and width is unchangeable.
    # Start with a common convolution layer.
    # Can choose whether adding Batch Normalization !
    def __init__(self, in_channels, out_channels, num_units, add_bn):
        super().__init__()
        if add_bn is False:
            self.layer = ConvLeaky(in_channels, out_channels, kernel_size=3, padding=1)
            self.block = ResBlock(out_channels, num_units, add_bn=False)
        elif add_bn is True:
            self.layer = ConvBNLeaky(in_channels, out_channels, kernel_size=3, padding=1)
            self.block = ResBlock(out_channels, num_units, add_bn=True)

    def forward(self, x):
        return self.block(self.layer(x))


class ResBlockHalf(nn.Module):
    # The height and width of the feature map will be HALF. The channel is CHANGEABLE.
    # Down sampling is implemented by convolution or maxpooling.
    # Can choose whether adding Batch Normalization !
    def __init__(self, in_channels, out_channels, half_type, num_units, add_bn):
        super().__init__()
        if half_type == 'convolution':
            #  this layer + the net convolution layer  will make the height and width of feature map be half.
            self.layer1 = nn.ZeroPad2d((1, 0, 1, 0))
            if add_bn is False:
                self.layer2 = ConvLeaky(in_channels, out_channels, kernel_size=3, stride=2)
                self.block = ResBlock(out_channels, num_units, add_bn=False)
            elif add_bn is True:
                self.layer2 = ConvBNLeaky(in_channels, out_channels, kernel_size=3, stride=2)
                self.block = ResBlock(out_channels, num_units, add_bn=True)

        elif half_type == 'max_pooling':
            self.layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
            if add_bn is False:
                self.layer2 = ConvLeaky(in_channels, out_channels, kernel_size=3, padding=1)
                self.block = ResBlock(out_channels, num_units, add_bn=False)
            elif add_bn is True:
                self.layer2 = ConvBNLeaky(in_channels, out_channels, kernel_size=3, padding=1)
                self.block = ResBlock(out_channels, num_units, add_bn=True)

    def forward(self, x):
        return self.block(self.layer2(self.layer1(x)))


class ResBlockTwice(nn.Module):
    # The height and width of feature map will be Twice. The channel is CHANGEABLE.
    # Up sampling is implemented by transposed convolution.
    # Can choose whether adding Batch Normalization !
    def __init__(self, in_channels, out_channels, twice_type, num_units,  add_bn):
        super().__init__()
        if twice_type == 'transposed_convolution':
            if add_bn is False:
                # a kernel size of 2 and a stride of 2 will increase the spatial dims by 2
                self.layer = TconvLeaky(in_channels, out_channels, kernel_size=2, stride=2)
                self.block = ResBlock(out_channels, num_units, add_bn=False)
            if add_bn is True:
                self.layer = TconvBNLeaky(in_channels, out_channels, kernel_size=2, stride=2)
                self.block = ResBlock(out_channels, num_units, add_bn=True)

    def forward(self, x):
        return self.block(self.layer(x))


# =====================================================================================
#                                  Complete Model
# =====================================================================================


class SMSFCFNet(nn.Module):
    """
        Inputs(1)
        ---------
            shape: N X 104

        Outputs(1)
        ----------
            shape: N x 1 X 64 X 64

    Note: N is the number of data samples
    """
    def __init__(self, add_bn):
        super().__init__()
        
        # ---------  Feature Extraction  -----------
        
        # prior layer
        if add_bn is False:
            self.pri_fc1 = FCLeaky(104, 1 * 16 * 64)
            self.pri_fc2 = FCLeaky(1 * 16 * 64, 1 * 32 * 64)
            self.pri_fc3 = FCLeaky(1 * 32 * 64, 1 * 64 * 64)
        elif add_bn is True:
            self.pri_fc1 = FCBNLeaky(104, 1 * 16 * 64)
            self.pri_fc2 = FCBNLeaky(1 * 16 * 64, 1 * 32 * 64)
            self.pri_fc3 = FCBNLeaky(1 * 32 * 64, 1 * 64 * 64)

        self.vresblock1 = ResBlockCC(1, 8, num_units=3, add_bn=add_bn)                                 # 8 x 64 x 64
        self.vresblock2 = ResBlockHalf(8, 16, half_type='convolution', num_units=3, add_bn=add_bn)     # 16 x 32 x 32
        self.vresblock3 = ResBlockHalf(16, 32, half_type='convolution', num_units=3, add_bn=add_bn)    # 32 x 16 x 16
        self.vresblock4 = ResBlockHalf(32, 64, half_type='convolution', num_units=7, add_bn=add_bn)    # 64 x 8 x 8
        self.vresblock5 = ResBlockHalf(64, 128, half_type='convolution', num_units=7, add_bn=add_bn)   # 128 x 4 x 4
        self.vresblock6 = ResBlockHalf(128, 256, half_type='convolution', num_units=5, add_bn=add_bn)  # 256 x 2 x 2

        # ----------  Feature Fusion  -----------
        
        if add_bn is False:
            self.s2FF64_1 = TconvLeaky(16, 8, kernel_size=2, stride=2)
            self.s2FF32_1 = TconvLeaky(32, 16, kernel_size=2, stride=2)
            self.s2FF16_1 = TconvLeaky(64, 32, kernel_size=2, stride=2)
            self.s2FF8_1 = TconvLeaky(128, 64, kernel_size=2, stride=2)
            self.s2FF4_1 = TconvLeaky(256, 128, kernel_size=2, stride=2)
        elif add_bn is True:
            self.s2FF64_1 = TconvBNLeaky(16, 8, kernel_size=2, stride=2)
            self.s2FF32_1 = TconvBNLeaky(32, 16, kernel_size=2, stride=2)
            self.s2FF16_1 = TconvBNLeaky(64, 32, kernel_size=2, stride=2)
            self.s2FF8_1 = TconvBNLeaky(128, 64, kernel_size=2, stride=2)
            self.s2FF4_1 = TconvBNLeaky(256, 128, kernel_size=2, stride=2)

        self.s2FF64_2 = ResBlock(8, num_units=3, add_bn=add_bn)
        self.s2FF32_2 = ResBlock(16, num_units=3, add_bn=add_bn)
        self.s2FF16_2 = ResBlock(32, num_units=3, add_bn=add_bn)
        self.s2FF8_2 = ResBlock(64, num_units=3, add_bn=add_bn)
        self.s2FF4_2 = ResBlock(128, num_units=3, add_bn=add_bn)

        # -------------  Output  -------------
        
        self.outconv = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, in_v):
        # Prior layers
        xv = self.pri_fc3(self.pri_fc2(self.pri_fc1(in_v)))

        # reshape to  1 x 64 x 64
        xv = xv.view(-1, 1, 64, 64)

        # features
        xv64 = self.vresblock1(xv)
        xv32 = self.vresblock2(xv64)
        xv16 = self.vresblock3(xv32)
        xv8 = self.vresblock4(xv16)
        xv4 = self.vresblock5(xv8)
        xv2 = self.vresblock6(xv4)

        # feature fusion to generate output image
        xf = self.s2FF4_2(xv4 + self.s2FF4_1(xv2))
        xf = self.s2FF8_2(xv8 + self.s2FF8_1(xf))
        xf = self.s2FF16_2(xv16 + self.s2FF16_1(xf))
        xf = self.s2FF32_2(xv32 + self.s2FF32_1(xf))
        xf = self.s2FF64_2(xv64 + self.s2FF64_1(xf))

        out_img = self.outconv(xf)

        return out_img


if __name__ == '__main__':
    
    # data
    in_v = torch.empty(3, 104)
    
    # model
    model = SMSFCFNet(add_bn=False)
    
    # forward
    out_img = model(in_v)

    # results
    print('----------------------------')
    print(in_v.shape)
    print(out_img.shape)
    print('----------------------------')




