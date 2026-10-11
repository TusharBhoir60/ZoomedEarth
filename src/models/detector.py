import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class BuildingDetectorUNet(nn.Module):
    """
    Compact U-Net architecture for ZoomedEarth Tier C Building Detection.
    Designed for RTX 3050 (6GB VRAM) running 256x256 or 512x512 windows.
    Parameter count: ~1.9M.
    """
    def __init__(self, in_channels=4, out_channels=1, base_filters=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters*2, base_filters*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters*4, base_filters*8))
        
        self.up1 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters*8, base_filters*4)
        
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters*4, base_filters*2)
        
        self.up3 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters*2, base_filters)
        
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Sanitize non-finite inputs defensively before convolutions
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)
        
        return self.outc(x)
