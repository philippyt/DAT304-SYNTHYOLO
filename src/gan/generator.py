import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetGenerator, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2))
        self.enc5 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2))
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2))

        self.dec5 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.Dropout(0.4), nn.ReLU(inplace=True))
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.Dropout(0.4), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.Dropout(0.2), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, apply_noise=True):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        if apply_noise:
            noise_level = 0.15 if self.training else 0.05
            b = b + noise_level * torch.randn_like(b)
        d5 = self.dec5(b)
        d5 = self.handle_size_mismatch(d5, e5)
        d5 = torch.cat([d5, e5], dim=1)
        d4 = self.dec4(d5)
        d4 = self.handle_size_mismatch(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d3 = self.dec3(d4)
        d3 = self.handle_size_mismatch(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d2 = self.dec2(d3)
        d2 = self.handle_size_mismatch(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d1 = self.dec1(d2)
        d1 = self.handle_size_mismatch(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        return self.final_layer(d1)

    def handle_size_mismatch(self, x, target):
        if x.size() != target.size():
            return nn.functional.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=False)
        return x