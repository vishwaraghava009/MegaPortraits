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
        self.initial = nn.Conv3d(248, 2048, kernel_size=1, stride=1)  # Adjusted input channels
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
        # Print shapes for debugging
        print(f"Original shapes - R: {R.shape}, t: {t.shape}, z: {z.shape}, e: {e.shape}")
        
        # Ensure all inputs have the correct shape
        batch_size = R.size(0)
        
        # Reshape R and t to match the expected dimensions
        R = R.view(batch_size, -1, 1, 1, 1)
        t = t.view(batch_size, -1, 1, 1, 1)
        
        # Ensure z and e have the correct shape: [batch_size, channels, depth, height, width]
        if len(z.size()) != 5 or len(e.size()) != 5:
            raise ValueError(f"Expected z and e to have 5 dimensions, got {z.size()} and {e.size()} instead.")
        
        # Print shapes after reshaping for debugging
        print(f"Reshaped - R: {R.shape}, t: {t.shape}, z: {z.shape}, e: {e.shape}")

        # Concatenate z and e along the channel dimension
        combined_ze = torch.cat((z, e), dim=1)
        
        # Print shape after concatenation for debugging
        print(f"Combined z and e shape: {combined_ze.shape}")
        
        # Ensure the combined tensor has the expected number of channels
        expected_channels = R.size(1) + t.size(1) + combined_ze.size(1)
        print(f"Expected channels after concatenation: {expected_channels}")
        
        # Concatenate R, t, and combined_ze along the channel dimension
        combined_rt = torch.cat((R, t), dim=1)
        combined_rt = combined_rt.expand(-1, -1, combined_ze.size(2), combined_ze.size(3), combined_ze.size(4))
        x = torch.cat((combined_rt, combined_ze), dim=1)
        
        # Print shape after concatenation for debugging
        print(f"Concatenated shape: {x.shape}")
        
        # Ensure the concatenated tensor has the expected number of channels (248)
        if x.size(1) != 248:
            raise ValueError(f"Concatenated tensor has {x.size(1)} channels, expected 248.")
        
        # Forward pass through the network
        x = F.relu(self.initial(x))
        x = self.res_blocks(x)
        x = self.final(x)
        return self.tanh(x)

if __name__ == "__main__":
    model = WarpingGenerator()
    print(model)
    batch_size = 7  # Example batch size
    R = torch.randn(batch_size, 6)  # Head pose representation (e.g., 6 DOF)
    t = torch.randn(batch_size, 50)  # Translation vectors (e.g., 50)
    z = torch.randn(batch_size, 96, 16, 28, 28)  # Appearance features (e.g., 96, 16, 28, 28)
    e = torch.randn(batch_size, 96, 16, 28, 28)  # Expression features (e.g., 96, 16, 28, 28)
    output = model(R, t, z, e)
    print(f'Output shape: {output.shape}')
