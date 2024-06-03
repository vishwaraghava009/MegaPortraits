import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        num_groups = min(32, out_channels // 4)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return F.relu(out)

class Conv3D(nn.Module):
    def __init__(self):
        super(Conv3D, self).__init__()
        self.res_blocks = nn.Sequential(
            ResidualBlock3D(192, 192),  # Corrected from 3 to 192 as per G3D
            nn.MaxPool3d((1, 2, 2)),
            ResidualBlock3D(192, 384),
            nn.MaxPool3d((2, 2, 2)),
            ResidualBlock3D(384, 512),
            ResidualBlock3D(512, 512),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(512, 384),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(384, 192),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(192, 96),
            nn.GroupNorm(num_groups=32, num_channels=96),
            nn.ReLU(),
            nn.Conv3d(96, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Final activation as Tanh
        )

    def forward(self, x):
        x = self.res_blocks(x)
        return x

if __name__ == "__main__":
    model = Conv3D()
    print(model)
