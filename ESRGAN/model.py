import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_act=False, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,**kwargs),
            nn.LeakyReLU(0.2) if use_act else nn.Identity()
        )
        self.use_act = use_act
    def forward(self, input):
        return self.block(input)

class DenseBlock(nn.Module):
    def __init__(self, in_channel, channel=32, num_conv=5, beta=0.2) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_conv = num_conv
        self.beta = beta
        for i in range(num_conv):
            self.layers.append(
                ConvBlock(
                    in_channel + i*channel, 
                    channel if i <= 3 else in_channel,
                    kernel_size=3, 
                    stride=1, 
                    padding=1,
                    use_act=True if i <= 3 else False
                )
            )
    def forward(self, input):
        cur_input = input
        out_i = None
        for i in range(self.num_conv):
            out_i = self.layers[i](cur_input)
            cur_input = torch.concat((cur_input, out_i), dim=1)
        return out_i * self.beta + input

class RRDB(nn.Module):
    def __init__(self, in_channel, num_dense=3, beta=0.2):
        super().__init__()
        self.beta = beta
        self.rrdb = nn.Sequential(
            *[DenseBlock(in_channel) for _ in range(num_dense)]
        )
    def forward(self, input):
        return self.rrdb(input) * self.beta + input

class Generator(nn.Module):
    def __init__(self, img_channel, num_channel=64, num_residuals=23):
        super().__init__()
        self.initial = nn.Conv2d(img_channel, num_channel, 3, 1, 1, bias=True)
        self.rrdbs = nn.Sequential(*[RRDB(num_channel) for _ in range(num_residuals)])
        self.conv = nn.Conv2d(img_channel, num_channel, 3, 1, 1, bias=True)
        self.upsample = nn.Sequential(
            nn.Upsample(num_channel),
            nn.Upsample(num_channel)
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channel, img_channel, 3, 1, 1, bias=True),
        )
    def forward(self, input):
        initial = self.initial(input)
        res = self.conv(self.rrdbs(initial)) + initial
        res = self.upsample(res)
        return self.final(res)

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