#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# baseline of hidden channels
encoder_layers = (16,32,64,128,256)
decoder_layers = (256,128,64,32,16)

class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, a_channels):
        """
        Params
        ======
        g_channels (int):
            channels of g
        
        x_channels (int):
            channels of x

        a_channels (int):
            channels of the images after addition
        """
        super().__init__()

        self.conv_g = nn.Sequential(
            nn.Conv2d(g_channels, a_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(a_channels)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(x_channels, a_channels, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(a_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(a_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.conv_g(g)
        x1 = self.conv_x(x)
        x1 = F.interpolate(x1, g1.shape[-2:], mode='bilinear')
        a = self.relu(g1 + x1)
        a = self.psi(a)
        a = F.interpolate(a, x.shape[-2:], mode='bilinear')        
        return x * a
                

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        
        channels = [in_channels] + list(hidden_channels)
        
        self.enc_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

        
    def forward(self, enc_features):
        enc_features = enc_features[::-1]
        for i in range(len(self.channels)-1):
            enc_feature = enc_features[i+1]
            x = self.up_convs[i](enc_features[i])            
            x = torch.cat([F.interpolate(x, enc_feature.shape[-2:], mode='bilinear'), enc_feature], dim=1)  
            x = self.dec_blocks[i](x)
        return x

class AttentionDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.att_blocks = nn.ModuleList([AttentionBlock(channels[i], channels[i+1], channels[i]) for i in range(len(channels)-1)])
        
    def forward(self, enc_features):
        enc_features = enc_features[::-1]
        for i in range(len(self.channels)-1):
            g, x = enc_features[i], enc_features[i+1]
            enc_feature = self.att_blocks[i](g, x)
            x = self.up_convs[i](g)
            x = torch.cat([F.interpolate(x, enc_feature.shape[-2:],mode='bilinear'), enc_feature], dim=1)  
            x = self.dec_blocks[i](x)
        return x    
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, scale=4):
        super().__init__()
        enc_channels = [scale * elem for elem in encoder_layers]
        dec_channels = [scale * elem for elem in decoder_layers]
        self.encoder = Encoder(in_channels, enc_channels)
        self.decoder = Decoder(dec_channels)
        self.head = nn.Conv2d(dec_channels[-1], out_channels, 1)

    def forward(self, x):
        enc_features = self.encoder(x)
        out = self.decoder(enc_features)
        out = self.head(out)

        return out

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, scale=4):
        super().__init__()
        enc_channels = [scale * elem for elem in encoder_layers]
        dec_channels = [scale * elem for elem in decoder_layers]
        self.encoder = Encoder(in_channels, enc_channels)
        self.decoder = AttentionDecoder(dec_channels)
        self.head = nn.Conv2d(dec_channels[-1], out_channels, 1)
        
    def forward(self, x):
        enc_features = self.encoder(x)
        out = self.decoder(enc_features)
        out = self.head(out)
        return out

    
if __name__ == '__main__':

    # # Test EncoderBlock
    # enc_block = Block(1, 64)
    # x = torch.randn(1, 1, 11, 11)
    # y = enc_block(x)
    # print(x.shape, y.shape)

    # # Test Encoder
    # encoder = Encoder(3)
    # x = torch.randn(1, 3, 400, 400)
    # encoder_features = encoder(x)
    # for feature in encoder_features:
    #     print(feature.shape)

    # # Test Decoder
    # decoder = Decoder()
    # x = torch.randn(1, 1024, 25, 25)
    # y = decoder(x, encoder_features)
    # print(y.shape)
    
    # nx, ny = 33, 33
    # print('\n####################')
    # print('      UNet')
    # print('####################')
    # x = torch.randn(4, 3, nx, ny)
    # model = UNet(scale=2)
    # y = model(x)    
    # print(f'input shape: {x.shape}, output shape: {y.shape}')

    print('\n####################')
    print('   AttentionUNet')
    print('####################')
    x = torch.randn(4, 1, 131, 131)
    model = AttentionUNet(1, 1)
    y = model(x)    
    print(f'input shape: {x.shape}, output shape: {y.shape}')
    
