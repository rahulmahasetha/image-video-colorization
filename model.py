import torch
import torch.nn as nn
#from model import UNetGenerator, PatchGANDiscriminator
# ---------- Generator: U-Net ----------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):  # predict ab channels
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Decoder
        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(512, 128)   # 512 from skip + 256
        self.dec2 = self.upconv_block(256, 64)
        self.dec1 = self.upconv_block(128, out_channels)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)       # 64
        e2 = self.enc2(e1)      # 128
        e3 = self.enc3(e2)      # 256
        e4 = self.enc4(e3)      # 512

        d4 = self.dec4(e4)      # 256
        d4 = torch.cat([d4, e3], dim=1)  # skip

        d3 = self.dec3(d4)      # 128
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)      # 64
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)      # 2 (ab channels)
        return d1

# ---------- Discriminator: PatchGAN ----------
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):  # input = grayscale + ab (3 channels)
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)