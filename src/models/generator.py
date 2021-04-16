import torch
import torch.nn as nn


class ResizeCNNGenerator(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResizeCNNGenerator, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        # encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder2 = nn.Sequential(    
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder3 = nn.Sequential(    
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder4 = nn.Sequential(    
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder5 = nn.Sequential(    
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'), # increase size twice
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder5 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'), # increase size twice
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, self.out_channel, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # forward pass for encoder
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = self.encoder4(out)
        out = self.encoder5(out)

        # forward pass for decoder
        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)
        out = self.decoder4(out)
        out = self.decoder5(out)
        return out