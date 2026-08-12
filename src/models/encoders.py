import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, base_filters=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, fusion_channels=64):
        super(FeatureFusion, self).__init__()
        self.conv_fuse = nn.Conv2d(in_channels, fusion_channels, 1)
        self.bn = nn.BatchNorm2d(fusion_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, optical, sar):
        # Concatenate optical and SAR features
        combined = torch.cat([optical, sar], dim=1)
        features = self.conv_fuse(combined)
        features = self.bn(features)
        features = self.relu(features)
        return features


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super(Decoder, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.up1 = nn.ConvTranspose2d(in_channels, base_filters, 2, stride=2)
        self.dec1 = conv_block(base_filters * 2, base_filters)
        
        self.up2 = nn.ConvTranspose2d(base_filters, base_filters // 2, 2, stride=2)
        self.dec2 = conv_block(base_filters, base_filters // 2)
        
        self.up3 = nn.ConvTranspose2d(base_filters // 2, base_filters // 4, 2, stride=2)
        self.dec3 = conv_block(base_filters // 2, base_filters // 4)
        
        self.up4 = nn.ConvTranspose2d(base_filters // 4, base_filters // 8, 2, stride=2)
        self.dec4 = conv_block(base_filters // 4 + base_filters // 8, base_filters // 8)
        
        self.final = nn.Conv2d(base_filters // 8, out_channels, 1)
        
    def forward(self, x, skip1, skip2, skip3, skip4):
        d1 = self.up1(x)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat([d4, skip4], dim=1)
        d4 = self.dec4(d4)
        
        out = self.final(d4)
        return out