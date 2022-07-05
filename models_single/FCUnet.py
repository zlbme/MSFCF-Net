"""
Implementation of Networks "FC-UNet" in paper:

Chen, Z., Yang, Y., Jia, J. and Bagnaninchi, P., 2020, May. Deep Learning Based Cell Imaging with Electrical Impedance
Tomography. In 2020 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) (pp. 1-6). IEEE.

Note: The only difference is that the output layer is not activated as we need to conduct quantitative image
      reconstruction.

Author: LIU Zhe
Date: 2020
"""
import torch
import torch.nn as nn


# ######################################################################################################################
#                                             Components of Model
# ######################################################################################################################

class FCReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class TwoConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvReLU(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class TconvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, output_padding=output_padding, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.tconv(x))


# ######################################################################################################################
#                                              The Full Model
# ######################################################################################################################

class FCUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = FCReLU(104, 1 * 64 * 64)

        # Left
        self.L_64 = TwoConvReLU(1, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.L_32 = TwoConvReLU(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.L_16 = TwoConvReLU(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottom
        self.B_8 = TwoConvReLU(256, 512)

        # Right
        self.tconv1 = TconvReLU(512, 512, kernel_size=2, stride=2)
        self.R_16 = TwoConvReLU(768, 256)

        self.tconv2 = TconvReLU(256, 256, kernel_size=2, stride=2)
        self.R_32 = TwoConvReLU(384, 128)

        self.tconv3 = TconvReLU(128, 128, kernel_size=2, stride=2)
        self.R_64 = TwoConvReLU(192, 64)

        # Output
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 64, 64)  # reshape to  1 x 64 x 64

        x64 = self.L_64(x)
        x = self.maxpool1(x64)

        x32 = self.L_32(x)
        x = self.maxpool2(x32)

        x16 = self.L_16(x)
        x = self.maxpool3(x16)

        x = self.B_8(x)
        x = self.R_16(torch.cat((x16, self.tconv1(x)), dim=1))
        x = self.R_32(torch.cat((x32, self.tconv2(x)), dim=1))
        x = self.R_64(torch.cat((x64, self.tconv3(x)), dim=1))

        x = self.outconv(x)

        return x


if __name__ == '__main__':

    x = torch.empty(2, 104)

    model = FCUNet()

    y = model(x)

    # results
    print('----------------------------')
    print(x.shape)
    print(y.shape)
    print('----------------------------')







