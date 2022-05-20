import torch
import torch.nn as nn
import torch.nn.functional as F

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
        min = torch.min(diff.view(diff.shape[0], -1), dim=1)[0]
        max = torch.max(diff.view(diff.shape[0], -1), dim=1)[0]

        min = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        max = max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)

        weight = self.alpha * (diff - min) / (max - min) + self.beta
        return torch.mean(torch.abs(weight * (inputs - targets)))
