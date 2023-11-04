""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules.utils import  _pair

def soft_pool2d(x,kernel_size=2,stride=2,force_inplace=False):
    kernel_size=_pair(kernel_size)
    stride=_pair(stride)
    _,c,h,w=x.size()
    e_x=torch.exp(x)
    return F.avg_pool2d(x.mul(e_x),kernel_size,stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x,kernel_size,stride=stride).mul_(sum(kernel_size)))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):

        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),

            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        #x=soft_pool2d(x)
        return self.maxpool_conv(x)
class Down_SOFT(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.AvgPool2d(2),
            nn.Conv2d(in_channels,in_channels,kernel_size=3, stride=2,padding=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        #x=soft_pool2d(x)
        return self.maxpool_conv(x)
class Down4_SOFT(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(

            nn.Conv2d(in_channels,in_channels,kernel_size=5, stride=4,padding=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)
            self.inception=nn.Sequential(
                Up_Inception_Resnet(out_channels,out_channels),
                Up_Inception_Resnet(out_channels, out_channels),
                Up_Inception_Resnet(out_channels, out_channels),
                Up_Inception_Resnet(out_channels, out_channels),
                Up_Inception_Resnet(out_channels, out_channels)
            )


    def forward(self, x1, x2):


        x1 = torch.cat([x2, x1], dim=1)

        x1=self.up(x1)
        x1=self.conv(x1)
        x1=self.inception(x1)
        return x1


class UpNoConcat(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):

        x1 = self.up(x1)

        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sig=nn.Sigmoid()
    def forward(self, x):

        return self.sig(self.conv(x))
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sig=nn.Tanh()
    def forward(self, x):

        return self.sig(self.conv(x))
class Up_Inception_Resnet(nn.Module):
    def __init__(self, in_size, outn,scale=0.2):
        super(Up_Inception_Resnet, self).__init__()
        self.scale = scale

        self.branch_0 = BasicConv2(in_size, outn, kernel_size=1, stride=1)

        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, outn, kernel_size=1, stride=1),
            BasicConv2(outn, outn, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2(outn, outn, kernel_size=(3,1), stride=1, padding=(1,0))
        )
        self.conv = nn.Conv2d(2*outn, outn, kernel_size=1, stride=1)

        self.relu = nn.LeakyReLU(inplace=False)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv(out)
        return self.relu(out * self.scale + x)

class BasicConv2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding=0):
        super(BasicConv2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
class Inception_Resnet_A(nn.Module):
    def __init__(self, in_size, scale=0.2):
        super(Inception_Resnet_A, self).__init__()
        self.scale = scale
        self.branch_0 = BasicConv2(in_size,32, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 32, kernel_size=1, stride=1),
            BasicConv2(32, 32, kernel_size=3, stride=1, padding=1)

        )
        self.branch_2 = nn.Sequential(
            BasicConv2(in_size,32, kernel_size=1, stride=1),
            BasicConv2(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d(96, 32, stride=1, kernel_size=1)

        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        out = torch.cat((x0,x1, x2), dim=1)
        out = self.conv(out)

        return self.relu(x + self.scale * out)

class Reduction_A(nn.Module):
    def __init__(self, in_size, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = BasicConv2(in_size, n, kernel_size=3, stride=2,padding=1)
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, k,kernel_size=1, stride=1),
            BasicConv2(k, l, kernel_size=3, stride=1, padding=1),
            BasicConv2(l, m, kernel_size=3, stride=2,padding=1)
        )
        self.branch_2 = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        return torch.cat((x0, x1, x2), dim=1)

class Inception_Resnet_B(nn.Module):
    def __init__(self, in_size, scale=0.2):
        super(Inception_Resnet_B, self).__init__()
        self.scale = scale
        self.branch_0 = BasicConv2(in_size, 32, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 32, kernel_size=1, stride=1),
            BasicConv2(32, 32, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2(32, 32, kernel_size=(7,1), stride=1, padding=(3,0)),

        )
        self.conv = nn.Conv2d(64, 32, kernel_size=1, stride=1)

        self.relu = nn.LeakyReLU(inplace=False)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv(out)
        return self.relu(out * self.scale + x)

class Reduction_B(nn.Module):
    def __init__(self, in_size):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            BasicConv2(in_size, 256, kernel_size=1, stride=1),
            BasicConv2(256, 384, kernel_size=3, stride=2)
        )
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 256, kernel_size=1, stride=1),
            BasicConv2(256, 288, kernel_size=3, stride=2)
        )
        self.branch_2 = nn.Sequential(
            BasicConv2(in_size, 256, kernel_size=1, stride=1),
            BasicConv2(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2(288, 320, kernel_size=3, stride=2)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x3=soft_pool2d(x,kernel_size=3,stride=2)
        return torch.cat((x0, x1, x2, x3), dim=1)

class Inception_Resnet_C(nn.Module):
    def __init__(self, in_size, scale=1.0, activation=False):
        super(Inception_Resnet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = BasicConv2(in_size, 32, kernel_size=1, stride=1)

        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 32, kernel_size=1, stride=1),
            BasicConv2(32, 32, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2(32, 32, kernel_size=(3,1), stride=1, padding=(1,0))
        )
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        self.relu = nn.LeakyReLU(inplace=False)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv(out)
        if self.activation:
            return self.relu(out * self.scale + x)
        return out * self.scale + x

class Inception_Resnet_L2(nn.Module):
    def __init__(self, input_n,first=False,scale=0.2):
        super(Inception_Resnet_L2, self).__init__()
        self.scale = scale
        self.first=first
        self.branch_0 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.LeakyReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),
            nn.Conv3d(1, 1, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(1),
            nn.LeakyReLU()

        )
        self.branch_2 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),
            nn.Conv3d(1, 1, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),
            nn.Conv3d(1, 1, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(1),
            nn.LeakyReLU()

        )


        self.conv4 = BasicConv2(3*input_n, input_n, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):

        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x0=torch.squeeze(x0,dim=1)

        x1 = torch.squeeze(x1, dim=1)

        x2 = torch.squeeze(x2, dim=1)

        out = torch.cat((x0, x1,x2), dim=1)

        if self.first==False:
            out=self.conv4(out)
        out=torch.unsqueeze(out,dim=1)
        if self.first==False:
            out=self.relu(x+0.2*out)

        return out

def flip(x, dim=3):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
class InceptionResUNet3D(nn.Module):
    def __init__(self,n_channels=4,bilinear=False):
        super(InceptionResUNet3D, self).__init__()
        self.n_channels = n_channels

        self.bilinear = bilinear

        self.conv00_1 = BasicConv2(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv00_2 = BasicConv2(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv0_0 = BasicConv2(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = BasicConv2(16, 32, kernel_size=3, stride=1, padding=1)
        self.convz_0 = BasicConv2(8, 32, kernel_size=3, stride=1, padding=1)
        self.convz_1 = BasicConv2(32, 64, kernel_size=3, stride=1, padding=1)

        self.conv16_16_1 = BasicConv2(16, 16, kernel_size=1, stride=1, padding=0)
        self.conv16_16_2 = BasicConv2(16, 16, kernel_size=1, stride=1, padding=0)
        self.conv16_16_3 = BasicConv2(16, 16, kernel_size=1, stride=1, padding=0)

        self.conv4=nn.Conv2d(96,32,kernel_size=1, stride=1, padding=0)
        self.conv5 = BasicConv2(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_1_0=nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_1_1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_1_2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_1_3 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        self.conv_1_4=nn.Conv2d(32,16,kernel_size=1,stride=1,padding=0)
        self.conv_1_5 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)

        self.inception_l2x0=Inception_Resnet_L2(4,first=True)
        self.inception_l2x = nn.Sequential(
            Inception_Resnet_L2(32),
            Inception_Resnet_L2(32),
            Inception_Resnet_L2(32),
            Inception_Resnet_L2(32),
            Inception_Resnet_L2(32)

        )

        self.inception_a = nn.Sequential(
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17)
        )
        self.inception_a1 = nn.Sequential(
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17)
        )
        self.inception_a2 = nn.Sequential(
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17),
            Inception_Resnet_A(32, scale=0.17)
        )
        self.inception_b = nn.Sequential(
            Inception_Resnet_B(32, scale=0.2),
            Inception_Resnet_B(32, scale=0.2),
            Inception_Resnet_B(32, scale=0.2),
            Inception_Resnet_B(32, scale=0.2),
            Inception_Resnet_B(32, scale=0.2)
        )

        self.inception_c = nn.Sequential(
            Inception_Resnet_C(64, scale=0.2),
            Inception_Resnet_C(64, scale=0.2),
            Inception_Resnet_C(64, scale=0.2),
            Inception_Resnet_C(64, scale=0.2),
            Inception_Resnet_C(64, scale=0.2)
        )

        self.reduction_a = nn.Sequential(

            Reduction_A(16,32,32,32,32)
        )
        self.reduction_b = nn.Sequential(

            Reduction_A(16, 16, 16, 8, 8)
        )


        self.down1 = Reduction_A(32, 32,32,16,16)

        self.down2 = Reduction_A(32, 32, 32, 16, 16)
        self.down3 = Reduction_A(64, 64,64,32,32)

        self.down4 = Reduction_A(128, 128,128,64,64)
        self.down5=Reduction_A(256, 256,256,128,128)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.denorm1 = nn.BatchNorm2d(256)

        self.up2 = Up(512, 128, bilinear)

        self.up3 = Up(256, 64, bilinear)

        self.up4 = Up(128, 32, bilinear)

        self.up5 = Up(64, 32, bilinear)
        self.up6 = Up(32, 16, bilinear)                #32,1
        self.out0 = OutConv(64, 16)
        self.out1 = OutConv(64, 16)
        self.out2 = OutConv(64, 16)
        self.out3 = OutConv(64, 16)
        self.out4 = OutConv(64, 1)
        self.outc1 = Out(64, 1)
        self.outc2 = Out(64, 1)


        self.relu=nn.LeakyReLU()



    def forward(self, X,Z):

        XH = X[:,:,:,:,:]
        XH = torch.squeeze(XH, dim=1)
        XH = self.conv0_0(XH)
        XH = self.conv0_1(XH)
        XH = torch.unsqueeze(XH, dim=1)

        XH = self.inception_l2x(XH)

        XH=torch.squeeze(XH,dim=1)


        Z=torch.squeeze(Z,dim=1)
        Z=self.convz_0(Z)
        Z = self.convz_1(Z)


        X2_0= torch.cat((XH,Z), dim=1)
        X0 = self.conv4(X2_0)
        
        X2_0=self.inception_a1(X0)
        X2_0=self.conv_1_1(X2_0)

        X2_1=flip(X0)

        X2_1=self.inception_a2(X2_1)
        X2_1=flip(X2_1)
        X2_1=self.conv_1_2(X2_1)


        h1=self.relu(X2_0+X2_1)     #600


        h2=self.down1(h1)
        h2=self.conv_1_5(h2)
        h2=self.inception_b(h2)

        h3 = self.down2(h2)

        h4 = self.down3(h3)
        h5 = self.down4(h4)

        h6 = self.down5(h5)


        h = F.leaky_relu(self.denorm1(self.up1(h6)))

        h = self.up2(h, h5)

        h = self.up3(h, h4)

        h = self.up4(h, h3)

        h = self.up5(h, h2)

        h=torch.cat((h,h1),dim=1)

        out4 = self.out4(h)     #mask
        logits0=self.outc1(h)   #cos2
        logits1 = self.outc2(h)   #sin2

        return out4,logits0,logits1

