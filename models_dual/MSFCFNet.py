# Multi-Scale Feature Cross Fusion Net (MSFCF-Net)
# @author: LIU Zhe
import torch
import torch.nn as nn
import math


# ============================================================================================
#                                    Model Components
# ============================================================================================

# ------------------------------------ Basic Modules -----------------------------------------


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

# ---------------------------------- Attention Modules ---------------------------------

# The channel attention and spatial attention mechanism below is proposed in the paper:
# "Woo, S., Park, J., Lee, J.Y. and So Kweon, I., 2018. Cbam: Convolutional block attention module. In Proceedings of
# the European conference on computer vision (ECCV) (pp. 3-19)."


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        # Layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # math.ceil makes the channel number non-zero
        self.conv1 = nn.Conv2d(in_channels, math.ceil(in_channels/ratio), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(math.ceil(in_channels/ratio), in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu1(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu1(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ---------------------------------- Residual  Modules -------------------------------------


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

# ----------------------------------  Multi modal Feature Fusion Modules ------------------------------------


class FeatureFusionUnit(nn.Module):
    def __init__(self, in_channels, attention_type, fusion_type, add_bn):
        """
        Note: if fusion_type == 'addition', output size is totally the SAME as input size;
              if fusion_type == 'concatenation', output channel is TWICE than input channel;
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.attention_type = attention_type

        if add_bn is False:
            # main features
            self.p1_layer1 = ConvLeaky(in_channels, math.ceil(in_channels / 2), kernel_size=1)
            self.p1_layer2 = ConvLeaky(math.ceil(in_channels / 2), math.ceil(in_channels / 2), kernel_size=3, padding=1)
            self.p1_layer3 = ConvLeaky(math.ceil(in_channels / 2), in_channels, kernel_size=1)
            # auxiliary features
            self.p2_layer1 = ConvLeaky(in_channels, math.ceil(in_channels / 2), kernel_size=1)
            self.p2_layer2 = ConvLeaky(math.ceil(in_channels / 2), math.ceil(in_channels / 2), kernel_size=3, padding=1)
            self.p2_layer3 = ConvLeaky(math.ceil(in_channels / 2), in_channels, kernel_size=1)

        elif add_bn is True:
            # main features
            self.p1_layer1 = ConvBNLeaky(in_channels, math.ceil(in_channels / 2), kernel_size=1)
            self.p1_layer2 = ConvBNLeaky(math.ceil(in_channels / 2), math.ceil(in_channels / 2), kernel_size=3, padding=1)
            self.p1_layer3 = ConvBNLeaky(math.ceil(in_channels / 2), in_channels, kernel_size=1)
            # auxiliary features
            self.p2_layer1 = ConvBNLeaky(in_channels, math.ceil(in_channels / 2), kernel_size=1)
            self.p2_layer2 = ConvBNLeaky(math.ceil(in_channels / 2), math.ceil(in_channels / 2), kernel_size=3, padding=1)
            self.p2_layer3 = ConvBNLeaky(math.ceil(in_channels / 2), in_channels, kernel_size=1)

        if attention_type == 'channel':
            self.p1_ca = ChannelAttention(in_channels, ratio=16)
            self.p2_ca = ChannelAttention(in_channels, ratio=16)
        elif attention_type == 'spatial':
            self.p1_sa = SpatialAttention(kernel_size=3)
            self.p2_sa = SpatialAttention(kernel_size=3)
        elif attention_type == 'channel_spatial':
            self.p1_ca = ChannelAttention(in_channels, ratio=4)
            self.p1_sa = SpatialAttention(kernel_size=3)
            self.p2_ca = ChannelAttention(in_channels, ratio=4)
            self.p2_sa = SpatialAttention(kernel_size=3)

    def forward(self, x1, x2):
        # x1 is the main feature
        # x2 is the auxiliary feature
        x1 = self.p1_layer3(self.p1_layer2(self.p1_layer1(x1)))
        x2 = self.p2_layer3(self.p2_layer2(self.p2_layer1(x2)))

        if self.fusion_type == 'addition':
            if self.attention_type == 'channel':
                x1 = self.p1_ca(x1) * x1
                x2 = self.p2_ca(x2) * x2
                return x1 + x2
            elif self.attention_type == 'spatial':
                x1 = self.p1_sa(x1) * x1
                x2 = self.p2_sa(x2) * x2
                return x1 + x2
            elif self.attention_type == 'channel_spatial':
                x1 = self.p1_ca(x1) * x1
                x1 = self.p1_sa(x1) * x1
                x2 = self.p2_ca(x2) * x2
                x2 = self.p2_sa(x2) * x2
                return x1 + x2

        elif self.fusion_type == 'concatenation':
            if self.attention_type == 'channel':
                x1 = self.p1_ca(x1) * x1
                x2 = self.p2_ca(x2) * x2
                return torch.cat((x1, x2), dim=1)
            elif self.attention_type == 'spatial':
                x1 = self.p1_sa(x1) * x1
                x2 = self.p2_sa(x2) * x2
                return torch.cat((x1, x2), dim=1)
            elif self.attention_type == 'channel_spatial':
                x1 = self.p1_ca(x1) * x1
                x1 = self.p1_sa(x1) * x1
                x2 = self.p2_ca(x2) * x2
                x2 = self.p2_sa(x2) * x2
                return torch.cat((x1, x2), dim=1)


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, attention_type, fusion_type, num_resunits, add_bn):
        super().__init__()
        self.block1 = FeatureFusionUnit(in_channels, attention_type, fusion_type, add_bn)
        if fusion_type == 'addition':
            self.block2 = ResBlock(in_channels, num_units=num_resunits, add_bn=add_bn)
        elif fusion_type == 'concatenation':
            self.block2 = ResBlockCC(2*in_channels, in_channels, num_units=num_resunits, add_bn=add_bn)

    def forward(self, x1, x2):
        # x1 is the main feature
        # x2 is the auxiliary feature
        return self.block2(self.block1(x1, x2))


# ============================================================================================
#                                     Complete Model
# ============================================================================================

class MSFCFNet(nn.Module):
    """
    Multi-Scale Feature Cross Fusion Net (MSFCF-Net)

    Model instantiation in the published paper:

              model = MSFCFNet(v_type='vector', msk_channel=1, add_bn=False)
    """

    def __init__(self, v_type, msk_channel, add_bn):
        super().__init__()
     
        # ----------------  Voltage Path  -------------------
       
        # Prior Layer
        if v_type == 'vector':
            if add_bn is False:
                self.pri_fc1 = FCLeaky(104, 1 * 16 * 64)
                self.pri_fc2 = FCLeaky(1 * 16 * 64, 1 * 32 * 64)
                self.pri_fc3 = FCLeaky(1 * 32 * 64, 1 * 64 * 64)
            elif add_bn is True:
                self.pri_fc1 = FCBNLeaky(104, 1 * 16 * 64)
                self.pri_fc2 = FCBNLeaky(1 * 16 * 64, 1 * 32 * 64)
                self.pri_fc3 = FCBNLeaky(1 * 32 * 64, 1 * 64 * 64)

        # Feature Extraction
        self.vresblock1 = ResBlockCC(1, 8, num_units=3, add_bn=add_bn)                                 # 8 x 64 x 64
        self.vresblock2 = ResBlockHalf(8, 16, half_type='convolution', num_units=3, add_bn=add_bn)     # 16 x 32 x 32
        self.vresblock3 = ResBlockHalf(16, 32, half_type='convolution', num_units=3, add_bn=add_bn)    # 32 x 16 x 16
        self.vresblock4 = ResBlockHalf(32, 64, half_type='convolution', num_units=7, add_bn=add_bn)    # 64 x 8 x 8
        self.vresblock5 = ResBlockHalf(64, 128, half_type='convolution', num_units=7, add_bn=add_bn)   # 128 x 4 x 4
        self.vresblock6 = ResBlockHalf(128, 256, half_type='convolution', num_units=5, add_bn=add_bn)  # 256 x 2 x 2
        
        # -------------------  Mask Path  --------------------
       
        if msk_channel == 1:
            self.mresblock1 = ResBlockCC(1, 8, num_units=3, add_bn=add_bn)  # 8 x 64 x 64
        elif msk_channel == 2:
            self.mresblock1 = ResBlockCC(2, 8, num_units=3, add_bn=add_bn)  # 8 x 64 x 64

        self.mresblock2 = ResBlockHalf(8, 16, half_type='convolution', num_units=3, add_bn=add_bn)     # 16 x 32 x 32
        self.mresblock3 = ResBlockHalf(16, 32, half_type='convolution', num_units=3, add_bn=add_bn)    # 32 x 16 x 16
        self.mresblock4 = ResBlockHalf(32, 64, half_type='convolution', num_units=7, add_bn=add_bn)    # 64 x 8 x 8
        self.mresblock5 = ResBlockHalf(64, 128, half_type='convolution', num_units=7, add_bn=add_bn)   # 128 x 4 x 4
        self.mresblock6 = ResBlockHalf(128, 256, half_type='convolution', num_units=5, add_bn=add_bn)  # 256 x 2 x 2
        
        # -------------------  Step 1: Dual Model Feature Fusion  -------------------
        
        self.s1FF64 = FeatureFusionBlock(8, attention_type='channel_spatial', fusion_type='concatenation', num_resunits=3, add_bn=add_bn)
        self.s1FF32 = FeatureFusionBlock(16, attention_type='channel_spatial', fusion_type='concatenation', num_resunits=3, add_bn=add_bn)
        self.s1FF16 = FeatureFusionBlock(32, attention_type='channel_spatial', fusion_type='concatenation', num_resunits=3, add_bn=add_bn)
        self.s1FF8 = FeatureFusionBlock(64, attention_type='channel', fusion_type='concatenation', num_resunits=3, add_bn=add_bn)
        self.s1FF4 = FeatureFusionBlock(128, attention_type='channel', fusion_type='concatenation', num_resunits=3, add_bn=add_bn)
        self.s1FF2 = FeatureFusionBlock(256, attention_type='channel', fusion_type='concatenation', num_resunits=3, add_bn=add_bn)
      
        # -------------------  Step 2: Multi-scale Feature Fusion  -------------------
        
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
        
        # ---------------------  Output  --------------------
       
        self.outconv = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, inV, inMsk):
        
       # --------------------- Voltage Path -----------------------
    
        xv = self.pri_fc3(self.pri_fc2(self.pri_fc1(inV)))
        # reshape to  1 x 64 x 64
        xv = xv.view(-1, 1, 64, 64)
        #
        xv64 = self.vresblock1(xv)
        xv32 = self.vresblock2(xv64)
        xv16 = self.vresblock3(xv32)
        xv8 = self.vresblock4(xv16)
        xv4 = self.vresblock5(xv8)
        xv2 = self.vresblock6(xv4)
        
        # --------------------- Mask Path -----------------------
        
        xm64 = self.mresblock1(inMsk)
        xm32 = self.mresblock2(xm64)
        xm16 = self.mresblock3(xm32)
        xm8 = self.mresblock4(xm16)
        xm4 = self.mresblock5(xm8)
        xm2 = self.mresblock6(xm4)
        
        # --------------- Feature Fusion to Generate Output Image -------------
        
        xf = self.s2FF4_2(self.s1FF4(xv4, xm4) + self.s2FF4_1(self.s1FF2(xv2, xm2)))
        xf = self.s2FF8_2(self.s1FF8(xv8, xm8) + self.s2FF8_1(xf))
        xf = self.s2FF16_2(self.s1FF16(xv16, xm16) + self.s2FF16_1(xf))
        xf = self.s2FF32_2(self.s1FF32(xv32, xm32) + self.s2FF32_1(xf))
        xf = self.s2FF64_2(self.s1FF64(xv64, xm64) + self.s2FF64_1(xf))
        outImg = self.outconv(xf)

        return outImg


if __name__ == '__main__':
     
    # data
    in_v = torch.empty(3, 104)
    in_msk = torch.empty(3, 1, 64, 64)

    # model
    model = MSFCFNet(v_type='vector', msk_channel=1, add_bn=False)

    # forward
    out_img = model(in_v, in_msk)

    # results
    print('----------------------------')
    print(in_v.shape)
    print(in_msk.shape)
    print(out_img.shape)
    print('----------------------------')








