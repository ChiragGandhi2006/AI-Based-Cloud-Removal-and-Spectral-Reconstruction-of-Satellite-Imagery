import torch
import torch.nn as nn


class AttentionUNetFull(nn.Module):
    """Complete Attention U-Net for cloud removal and spectral reconstruction."""
    
    def __init__(self, in_channels=13, out_channels=13, base_filters=32, 
                 use_sar=True, use_mask=True):
        super(AttentionUNetFull, self).__init__()
        
        self.use_sar = use_sar
        self.use_mask = use_mask
        
        # Calculate effective input channels
        effective_in = in_channels
        if use_mask:
            effective_in += 1  # mask channel
        if use_sar:
            effective_in += 2  # VV, VH
            
        # Initial convolution
        self.init_conv = nn.Conv2d(effective_in, base_filters, 3, padding=1)
        self.init_bn = nn.BatchNorm2d(base_filters)
        self.init_relu = nn.ReLU(inplace=True)
        
        # Encoder blocks
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = self._make_enc_block(base_filters, base_filters * 2)
        self.enc2 = self._make_enc_block(base_filters * 2, base_filters * 4)
        self.enc3 = self._make_enc_block(base_filters * 4, base_filters * 8)
        self.enc4 = self._make_enc_block(base_filters * 8, base_filters * 16)
        
        # Bottleneck
        self.bottleneck = self._make_enc_block(base_filters * 16, base_filters * 32)
        
        # Decoder blocks with attention
        self.up4 = nn.ConvTranspose2d(base_filters * 32, base_filters * 16, 2, stride=2)
        self.dec4 = self._make_dec_block(base_filters * 32, base_filters * 16)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec3 = self._make_dec_block(base_filters * 16, base_filters * 8)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec2 = self._make_dec_block(base_filters * 8, base_filters * 4)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec1 = self._make_dec_block(base_filters * 4, base_filters * 2)
        
        # Final output
        self.final = nn.Conv2d(base_filters * 2, out_channels, 1)
        
    def _make_enc_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_dec_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, sar=None, mask=None):
        # Concatenate auxiliary inputs if used
        if sar is not None and self.use_sar:
            x = torch.cat([x, sar], dim=1)
        if mask is not None and self.use_mask:
            x = torch.cat([x, mask], dim=1)

        # Initial processing
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        
        # Save skip connections
        s1 = x
        e1 = self.enc1(x)
        
        s2 = self.pool(e1)
        e2 = self.enc2(s2)
        
        s3 = self.pool(e2)
        e3 = self.enc3(s3)
        
        s4 = self.pool(e3)
        e4 = self.enc4(s4)
        
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out