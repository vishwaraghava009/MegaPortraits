import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def save_tensor(tensor, filename):
    os.makedirs('intermediate_tensors', exist_ok=True)
    torch.save(tensor, os.path.join('intermediate_tensors', filename))

def load_tensor(filename):
    return torch.load(os.path.join('intermediate_tensors', filename))

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

    def forward(self, x, idx):
        identity = self.skip(x)
        print(f"Memory before conv1: {torch.cuda.memory_allocated() / 1e9} GB")  # Debugging
        out = self.conv1(x)
        print(f"Memory after conv1: {torch.cuda.memory_allocated() / 1e9} GB")   # Debugging
        save_tensor(out, f'conv1_out_{idx}.pt')
        del out
        torch.cuda.empty_cache()
        
        out = load_tensor(f'conv1_out_{idx}.pt')
        out = F.relu(self.gn1(out))
        print(f"Memory after gn1: {torch.cuda.memory_allocated() / 1e9} GB")     # Debugging
        save_tensor(out, f'gn1_out_{idx}.pt')
        del out
        torch.cuda.empty_cache()

        out = load_tensor(f'gn1_out_{idx}.pt')
        out = self.conv2(out)
        print(f"Memory after conv2: {torch.cuda.memory_allocated() / 1e9} GB")   # Debugging
        save_tensor(out, f'conv2_out_{idx}.pt')
        del out
        torch.cuda.empty_cache()

        out = load_tensor(f'conv2_out_{idx}.pt')
        out = self.gn2(out)
        print(f"Memory after gn2: {torch.cuda.memory_allocated() / 1e9} GB")     # Debugging
        out += identity
        return F.relu(out)

class Conv3D(nn.Module):
    def __init__(self):
        super(Conv3D, self).__init__()
        self.res_blocks = nn.Sequential(
            ResidualBlock3D(3, 192),  # Adjusted input channels to match WarpingGenerator output
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
            nn.Tanh()
        )

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debugging statement
        for i, layer in enumerate(self.res_blocks):
            print(f"Before layer {i} ({layer.__class__.__name__}): shape = {x.shape}, memory = {torch.cuda.memory_allocated() / 1e9} GB")
            x = layer(x, i) if isinstance(layer, ResidualBlock3D) else layer(x)
            save_tensor(x, f'conv3d_layer_{i}.pt')
            del x
            torch.cuda.empty_cache()
            x = load_tensor(f'conv3d_layer_{i}.pt')
            print(f"After layer {i} ({layer.__class__.__name__}): shape = {x.shape}, memory = {torch.cuda.memory_allocated() / 1e9} GB")
        return x

if __name__ == "__main__":
    model = Conv3D()
    print(model)
    test_input = torch.randn(1, 3, 64, 224, 224)  # Example input
    test_output = model(test_input)
    print(f'Output shape: {test_output.shape}')
