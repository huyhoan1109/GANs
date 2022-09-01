import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_act=True, use_disc=False, use_bn=True, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, bias=not use_bn, **kwargs),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Identity(),
        )
        self.act = nn.LeakyReLU(0.2) if use_disc else nn.PReLU(out_channel)
        self.use_act = use_act
    def forward(self, input):
        if self.use_act:
            return self.act(self.block(input))
        else:
            return self.block(input)

class UpSampleBlock(nn.Module):
    def __init__(self, channel, scale_factor=2) -> None:
        super().__init__()
        # C x H x W => C * (scale^2) x H x W => C x (scale * H) x (scale * W)  
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), 
            nn.PReLU(channel) 
        )
    def forward(self, input):
        return self.block(input)

class ResidualBlock(nn.Module):
    def __init__(self, channel, **kwargs):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBlock(channel, channel, **kwargs),
            ConvBlock(channel, channel, use_act=False, **kwargs)
        )   
    def forward(self, input):
        return input + self.residual(input)

class Generator(nn.Module):
    def __init__(self, img_channel=3, num_residuals=16):
        super().__init__()
        self.initial = ConvBlock(img_channel, 64, use_bn=False, kernel_size=9, stride=1, padding=4)
        self.residual = nn.Sequential(
            *[ResidualBlock(64, kernel_size=3, stride=1, padding=1) for _ in range(num_residuals)]
        )
        self.conv = ConvBlock(64, 64, use_act=False, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Sequential(
            UpSampleBlock(64),
            UpSampleBlock(64)
        )
        self.final = ConvBlock(64, img_channel, use_act=False, use_bn=False, kernel_size=9, stride=1, padding=4)

    def forward(self, input):
        initial = self.initial(input)
        res = self.residual(initial)
        res = self.conv(res) + initial
        res = self.upsample(res)
        return torch.tanh(self.final(res))

class Discriminator(nn.Module):
    def __init__(self, in_channel, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        layers = []
        for idx, feature in enumerate(features):
            layers.append(
                ConvBlock(
                    in_channel,
                    feature,
                    use_disc=True,
                    use_bn=False if idx==0 else True,
                    kernel_size=3,
                    stride=1+idx%2,
                    padding=1
                )
            )
            in_channel = feature
        self.block = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    def forward(self, input):
        res = self.block(input)
        return torch.sigmoid(self.classifier(res))