from multiprocessing import reduction
from re import A
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_init):
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(F_g, F_init, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_init)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, F_init, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_init)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_init, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super(AttUNet, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = conv_block(in_c, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])
        self.conv4 = conv_block(filters[2], filters[3])
        self.conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_init=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_init=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_init=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_init=16)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.conv = nn.Conv2d(filters[0], out_c, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        e1 = self.conv1(x)

        e2 = self.maxpool1(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool2(e2)
        e3 = self.conv3(e3)
        
        e4 = self.maxpool3(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool4(e4)
        e5 = self.conv5(e5)

        d4 = self.Up5(e5)
        x4 = self.Att5(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv5(d4)

        d3 = self.Up4(d4)
        x3 = self.Att4(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv4(d3)

        d2 = self.Up3(d3)
        x2 = self.Att3(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv3(d2)

        d1 = self.Up2(d2)
        x1 = self.Att2(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_conv2(d1)

        out = self.conv(d1)

        return out
        
       
        
class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, polling=True):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    
    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        return self.encode(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.GroupNorm(32, middle_channels),
            nn.GELU(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        ]
        self.decode = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decode(x)
    
class UNet(nn.Module):
    
    def __init__(self, out_channels=1, in_channels=3, factor=1):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(in_channels, 32*factor, polling=False)
        self.enc2 = EncoderBlock(32*factor, 64*factor)
        self.enc3 = EncoderBlock(64*factor, 128*factor)        
        self.enc4 = EncoderBlock(128*factor, 256*factor)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = DecoderBlock(256*factor, 512*factor, 256*factor)
        self.dec4 = DecoderBlock(512*factor, 256*factor, 128*factor)
        self.dec3 = DecoderBlock(256*factor, 128*factor, 64*factor)        
        self.dec2 = DecoderBlock(128*factor, 64*factor, 32*factor)        
        self.dec1 = DecoderBlock(64*factor, 32*factor, 32*factor)
        self.final = nn.Conv2d(32*factor, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.pool(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class AttentionLoss(nn.Module):
    def __init__(self, N, diff_func=F.l1_loss):
        super().__init__()
        # self.lamda = torch.ones((N, N), requires_grad=True)
        self.register_parameter(name='lamda', param=torch.nn.Parameter(torch.ones((N, N))))
        self.active = nn.Sigmoid()
        self.diff_func = diff_func

    def forward(self, x, y):
        diff = self.diff_func(x, y, reduction='none')
        lamda = 100 * self.active(-0.05 * (self.lamda - 5))
        lamda = lamda.type_as(x)
        diff = diff * lamda
        
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!{self.lamda[0,0]:.6f}')
        return torch.mean(torch.abs(diff))

class WeightedLoss(torch.nn.Module):

    def __init__(self, diff_fun, alpha=10.0, beta=0.0):
        super(WeightedLoss, self).__init__()
        self.diff_fun = diff_fun
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        """
        'inputs' -> Hard BC predicted u,
        'targets' -> zero matrix.
        """
        diff = self.diff_fun(inputs, targets, reduction='none').detach()
        min = torch.min(diff.reshape(diff.shape[0], -1), dim=1)[0]
        max = torch.max(diff.reshape(diff.shape[0], -1), dim=1)[0]

        min = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        max = max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)

        weight = self.alpha * (diff - min) / (max - min) + self.beta
        return torch.mean(torch.abs(weight * (inputs - targets)))

class TruncateLoss(torch.nn.Module):
    
    def __init__(self, diff_fun, r=0.5):
        super(TruncateLoss, self).__init__()
        self.diff_fun = diff_fun
        self.r = r

    def forward(self, inputs, targets):
        diff = self.diff_fun(inputs, targets, reduction='none').detach()
        min = torch.min(diff.view(diff.shape[0], -1), dim=1)[0]
        max = torch.max(diff.view(diff.shape[0], -1), dim=1)[0]

        min = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        max = max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)

        weight = (diff - min)/(max - min)
        weight = (weight > self.r).float()
        weight = weight/self.r

        return torch.mean(torch.abs(weight * (inputs - targets)))

if __name__ == '__main__':
    x = torch.rand((4,3,400,400)).requires_grad_()
    # y = torch.zeros((4, 1, 400, 400)).requires_grad_()

    # net = AttUNet()
    net = UNet()
    y = net(x)
    print(y.shape)