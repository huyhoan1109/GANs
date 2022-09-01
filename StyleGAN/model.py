from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class WsLinear(nn.Module):
    def __init__(self, in_features, out_features, gain=2):
        super(WsLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (gain / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return self.linear(x * self.scale) + self.bias

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WsLinear(z_dim, w_dim),
            nn.ReLU(),
            WsLinear(w_dim, w_dim),
            nn.ReLU(),
            WsLinear(w_dim, w_dim),
            nn.ReLU(),
            WsLinear(w_dim, w_dim),
            nn.ReLU(),
            WsLinear(w_dim, w_dim),
            nn.ReLU(),
            WsLinear(w_dim, w_dim),
            nn.ReLU(),
            WsLinear(w_dim, w_dim),
            nn.ReLU(),
            WsLinear(w_dim, w_dim),
        )
    
    def forward(self, x):
        return self.mapping(x)

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WsLinear(w_dim, channels)
        self.style_bias = WsLinear(w_dim, channels)
    
    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return x * style_scale + style_bias

class WsConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WsConv2d(in_channels, out_channels)
        self.conv2 = WsConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.noise_inject1 = NoiseInjection(out_channels)
        self.noise_inject2 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)
    
    def forward(self, x, w):
        res = self.conv1(x)
        res = self.noise_inject1(res)
        res = self.leaky(res)
        res = self.adain1(res, w)
        res = self.conv2(res)
        res = self.noise_inject2(res)
        res = self.leaky(res)
        res = self.adain2(res, w)
        return res

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WsConv2d(in_channels, out_channels)
        self.conv2 = WsConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        res = self.conv1(x)
        res = self.leaky(res)
        res = self.conv2(res)
        res = self.leaky(res)
        return res

class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.start_const = nn.Parameter(torch.ones((1, in_channels, 4,  4)))
        self.map = MappingNetwork(z_dim, w_dim)
        self.initial_adain1 = AdaIN(in_channels, w_dim)
        self.initial_adain2 = AdaIN(in_channels, w_dim)
        self.initial_noise1 = NoiseInjection(in_channels)
        self.initial_noise2 = NoiseInjection(in_channels)
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.initial_rgb = WsConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks = nn.ModuleList()
        self.rgb_layers = nn.ModuleList([self.initial_rgb])

        for i in range(len(factors)-1):
            conv_in_g = int(in_channels * factors[i])
            conv_out_g = int(in_channels * factors[i+1])
            self.prog_blocks.append(GenBlock(conv_in_g, conv_out_g, w_dim))
            self.rgb_layers.append(
                WsConv2d(conv_out_g, img_channels, kernel_size=1, stride=1, padding=0)
            )
    
    
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)
    
    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        x = self.initial_adain1(self.initial_noise1(self.start_const), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

        if steps == 0:
            return self.initial_rgb(x)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
            out = self.prog_blocks[step](upscaled, w)
        
        final_upscaled = self.rgb_layers[steps-1](upscaled)
        final_out = self.rgb_layers[steps](out)
        
        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks = nn.ModuleList()
        self.rgb_layers = nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)
        for i in range(len(factors)-1, 0, -1):
            conv_in_d = int(in_channels * factors[i])
            conv_out_d = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_d, conv_out_d))
            self.rgb_layers.append(
                WsConv2d(img_channels, conv_in_d, kernel_size=1, stride=1, padding=0)
            )
        self.intial_rgb = WsConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.intial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WsConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WsConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WsConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )
    
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1-alpha) * downscaled
    
    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)
    
    def forward(self, x, alpha, steps):

        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)