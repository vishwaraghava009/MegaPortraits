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

class WarpingGenerator(nn.Module):
    def __init__(self):
        super(WarpingGenerator, self).__init__()
        self.initial = nn.Conv3d(8, 2048, kernel_size=1, stride=1)  # Changed input channels to match R, t, z, e
        self.res_blocks = nn.Sequential(
            ResidualBlock3D(2048, 512),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(512, 256),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(256, 128),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(128, 64),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(64, 32),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(32, 16),
        )
        self.final = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, R, t, z, e):
        x = torch.cat((R, t, z, e), dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # singleton dimensions for depth, height, width
        x = F.relu(self.initial(x))
        x = self.res_blocks(x)
        x = self.final(x)
        return self.tanh(x)

if __name__ == "__main__":
    model = WarpingGenerator()
    print(model)
    R = torch.randn(1, 2, 1, 1, 1)
    t = torch.randn(1, 2, 1, 1, 1)
    z = torch.randn(1, 2, 1, 1, 1)
    e = torch.randn(1, 2, 1, 1, 1)
    output = model(R, t, z, e)
    print(f'Output shape: {output.shape}')
