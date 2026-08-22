import torch
import torch.nn as nn

from .unet import UNet


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet(UNet):
    def __init__(self, in_channels=13, out_channels=13, base_filters=32, use_sar=True, use_mask=False):
        self.use_sar = use_sar
        self.use_mask = use_mask

        effective_in = in_channels
        if use_sar:
            effective_in += 2
        if use_mask:
            effective_in += 1

        super(AttentionUNet, self).__init__(effective_in, out_channels, base_filters)

        self.att4 = AttentionBlock(F_g=base_filters * 8, F_l=base_filters * 8, F_int=base_filters * 4)
        self.att3 = AttentionBlock(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2)
        self.att2 = AttentionBlock(F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters)
        self.att1 = AttentionBlock(F_g=base_filters, F_l=base_filters, F_int=base_filters // 2)

    def forward(self, x, sar=None, mask=None):
        if sar is not None and self.use_sar:
            x = torch.cat([x, sar], dim=1)
        if mask is not None and self.use_mask:
            x = torch.cat([x, mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, self.att4(d4, e4)], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.att3(d3, e3)], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.att2(d2, e2)], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.att1(d1, e1)], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out