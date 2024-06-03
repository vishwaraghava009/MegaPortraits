import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        num_groups = min(32, out_channels // 4)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return F.relu(out)

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

class AppearanceEncoder(nn.Module):
    def __init__(self):
        super(AppearanceEncoder, self).__init__()
        self.initial = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)  # Changed stride to 1
        self.res_blocks = nn.Sequential(
            ResidualBlock2D(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock2D(128, 256),
            nn.AvgPool2d(2),
            ResidualBlock2D(256, 512),
            nn.AvgPool2d(2),
            ResidualBlock2D(512, 1536)  # Corrected the final channel size to 1536
        )
        self.reshaped = nn.Sequential(
            nn.Conv2d(1536, 1536, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=1536)  # Ensuring divisible by 32
        )
        self.res3d_blocks = nn.Sequential(
            ResidualBlock3D(96, 96),
            ResidualBlock3D(96, 96),
            ResidualBlock3D(96, 96)
        )

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.res_blocks(x)
        x = F.relu(self.reshaped(x))
        batch_size, channels, height, width = x.shape
        new_channels = 96  # Corrected to 96
        depth = channels // new_channels
        x = x.view(batch_size, new_channels, depth, height, width)  # Reshape to 3D features
        x = self.res3d_blocks(x)
        return x

if __name__ == "__main__":
    model = AppearanceEncoder()
    print(model)
    test_input = torch.randn(1, 3, 224, 224)
    test_output = model(test_input)
    print(f'Output shape: {test_output.shape}')
