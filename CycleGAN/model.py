import torch
import torch.nn as nn

# implement Discriminator
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)   
        )
        layers = []
        in_c = features[0]
        for feature in features[1: ]:
            layers.append(
                Block(in_c, feature, stride=1 if feature==features[-1] else 2)
            )
            in_c = feature
        layers.append(nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        res = self.initial(input)
        return torch.sigmoid(self.model(res))

# Implement Generator
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs) 
            if down else
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    def forward(self, input):
        return self.conv(input)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, features=[64, 128, 256, 512], num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, features[0], kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        down_layers = []
        for i in range(0, len(features)-1):
            down_layers.append(
                ConvBlock(features[i], features[i+1], kernel_size=3, stride=2, padding=1)
            )
        self.down = nn.Sequential(*down_layers)
        self.residual = nn.Sequential(
            *[ResidualBlock(features[-1]) for _ in range(num_residuals)]
        )
        up_layers = []
        for i in range(len(features)-1, 0, -1):
            up_layers.append(
                ConvBlock(features[i], features[i-1], down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
        self.up = nn.Sequential(*up_layers)
        self.last = nn.Conv2d(features[0], img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
    def forward(self, input):
        res = self.initial(input)
        res = self.down(res)
        res = self.residual(res)
        res = self.up(res)
        return torch.tanh(self.last(res))