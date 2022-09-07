import torch
import torch.nn as nn
import torchvision.transforms.functional as fn

class DoubleConv(nn.Module):
    def __init__(self, inc, outc, padding='same'):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(3, 3), stride=(1,1), padding=padding, bias=True),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=(3, 3), stride=(1,1), padding='same', bias=True),
            nn.BatchNorm2d(outc),
            nn.ReLU()]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

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
    def __init__(self, in_c=3, out_c=1, features=16, bc=False):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dconv0 = DoubleConv(in_c, features)
        self.dconv1 = DoubleConv(features    , features * 2)
        self.dconv2 = DoubleConv(features * 2, features * 4)
        self.dconv3 = DoubleConv(features * 4, features * 8)
        self.dconv4 = DoubleConv(features * 8, features * 16)

        self.uconv4 = DoubleConv(features *16, features * 8)
        self.uconv3 = DoubleConv(features * 8, features * 4)
        self.uconv2 = DoubleConv(features * 4, features * 2)
        self.uconv1 = DoubleConv(features * 2, features)
        if bc:
            self.final = nn.Conv2d(features, out_c, (3, 3), (1, 1), 'same')
        else:
            self.final = nn.Conv2d(features, out_c, (3, 3), (1, 1), 'valid')

        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, (2, 2), (2, 2))
        self.up3 = nn.ConvTranspose2d(features * 8,  features * 4, (2, 2), (2, 2))
        self.up2 = nn.ConvTranspose2d(features * 4,  features * 2, (2, 2), (2, 2))
        self.up1 = nn.ConvTranspose2d(features * 2,  features * 1, (3, 3), (2, 2))

        self.ag1 = Attention_block(features * 8, features * 8, features * 4)
        self.ag2 = Attention_block(features * 4, features * 4, features * 2)
        self.ag3 = Attention_block(features * 2, features * 2, features * 1)
        self.ag4 = Attention_block(features, features, features)


    def forward(self, x):
        x1 = self.dconv0(x)
        x2 = self.dconv1(self.maxpool(x1))
        x3 = self.dconv2(self.maxpool(x2)) 
        x4 = self.dconv3(self.maxpool(x3)) 
        x5 = self.dconv4(self.maxpool(x4)) 
        y = self.up4(x5)
        y = self.uconv4(torch.cat([self.ag1(g=y, x=x4), y], 1))
        y = self.up3(y)
        y = self.uconv3(torch.cat([self.ag2(g=y, x=x3), y], 1))
        y = self.up2(y)
        y = self.uconv2(torch.cat([self.ag3(g=y, x=x2), y], 1))
        y = self.up1(y)
        y = self.uconv1(torch.cat([self.ag4(g=y, x=x1), y], 1))
        y = self.final(y)
        return y


class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, features=16, bc=False):
        '''
        mode1 ---> F1, mode2 ---> F2
        '''
        super().__init__()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

        self.dconv0 = DoubleConv(in_c, features)
        self.dconv1 = DoubleConv(features, features * 2)
        self.dconv2 = DoubleConv(features * 2, features * 4)
        self.dconv3 = DoubleConv(features * 4, features * 8)
        self.dconv4 = DoubleConv(features * 8, features * 16)
        
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, (2, 2), (2, 2))
        self.up3 = nn.ConvTranspose2d(features * 8,  features * 4, (2, 2), (2, 2))
        self.up2 = nn.ConvTranspose2d(features * 4,  features * 2, (2, 2), (2, 2))
        self.up1 = nn.ConvTranspose2d(features * 2,  features * 1, (3, 3), (2, 2))

        self.uconv3 = DoubleConv(features *16, features * 8)
        self.uconv2 = DoubleConv(features * 8, features * 4)
        self.uconv1 = DoubleConv(features * 4, features * 2)
        self.uconv0 = DoubleConv(features * 2, features * 1)

        if bc:
            self.final = nn.Conv2d(features, out_c, (3, 3), (1, 1), 'same')
        else:
            self.final = nn.Conv2d(features, out_c, (3, 3), (1, 1), 'valid')

    def forward(self , x):
        x0 = self.dconv0(x)
        x1 = self.dconv1(self.maxpool(x0))
        x2 = self.dconv2(self.maxpool(x1))
        x3 = self.dconv3(self.maxpool(x2))
        x4 = self.dconv4(self.maxpool(x3))

        y = self.uconv3(torch.cat([self.up4(x4),x3], 1))
        y = self.uconv2(torch.cat([self.up3(y), x2], 1))
        y = self.uconv1(torch.cat([self.up2(y), x1], 1))
        y = self.uconv0(torch.cat([self.up1(y), x0], 1))
        y = self.final(y)
        return y

# Do not delete it
model_names = {'UNet': UNet, 'AttUNet': AttUNet}

if __name__ == '__main__':
    x = torch.rand(4, 3, 65, 65)
    net1 = UNet(3, 1, 2)
    net2 = AttUNet(3, 1, 2)

    print(net1(x).shape)
    print(net2(x).shape)
