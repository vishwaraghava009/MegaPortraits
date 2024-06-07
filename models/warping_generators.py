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
        self.conv_initial = nn.Conv3d(256, 2048, kernel_size=1, stride=1)  # Corrected input channels to match the sum of inputs
        self.reshape = nn.Sequential(
            nn.Conv3d(2048, 2048, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=2048),
            nn.ReLU(),
            nn.Conv3d(2048, 512, kernel_size=3, stride=1, padding=1)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock3D(512, 256),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(256, 128),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(128, 64),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(64, 32)
        )
        self.final_conv = nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_initial(x)
        x = self.reshape(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return torch.tanh(x)  # Use Tanh to ensure output is in the range [-1, 1]

if __name__ == "__main__":
    model = WarpingGenerator()
    print(model)
    test_input = torch.randn(1, 256, 16, 16, 16)  # Example input
    test_output = model(test_input)
    print(f'Output shape: {test_output.shape}')
