import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os

intermediate_path = "/content/drive/MyDrive/VASA-1-master/intermediate_chunks"

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

        # Debugging memory before res_blocks
        print(f"Memory before res_blocks: {torch.cuda.memory_allocated() / 1e9} GB")

        # Process res_blocks in chunks
        chunk_size_res = 1  # Adjust chunk size as needed
        total_res_blocks_output = []
        for res_chunk_idx in range(0, x.size(0), chunk_size_res):
            res_chunk = x[res_chunk_idx:res_chunk_idx + chunk_size_res]
            res_chunk_output = self.process_res_blocks_in_chunks(res_chunk, res_chunk_idx)
            total_res_blocks_output.append(res_chunk_output.cpu())
            
            del res_chunk, res_chunk_output
            torch.cuda.empty_cache()
            gc.collect()

        x = torch.cat(total_res_blocks_output, dim=0).to(x.device)
        
        # Debugging memory after res_blocks
        print(f"Memory after res_blocks: {torch.cuda.memory_allocated() / 1e9} GB")

        x = self.final(x)
        
        # Debugging memory after final layer
        print(f"Memory after final layer: {torch.cuda.memory_allocated() / 1e9} GB")
        
        return self.tanh(x)

    def process_res_blocks_in_chunks(self, x, res_chunk_idx):
        """
        Process the res_blocks in chunks to manage memory usage better.
        """
        chunk_size = 1  # Adjust chunk size as needed
        res_block_outputs = []
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i:i+chunk_size]
            for block_idx, block in enumerate(self.res_blocks):
                x_chunk = self.process_block_in_chunks(block, x_chunk, res_chunk_idx, block_idx)
            res_block_outputs.append(x_chunk.cpu())
            del x_chunk
            torch.cuda.empty_cache()
            gc.collect()

        return torch.cat(res_block_outputs, dim=0).to(x.device)

    def process_block_in_chunks(self, block, x, res_chunk_idx, block_idx):
        """
        Process each block within res_blocks in chunks.
        """
        chunk_size = 1  # Adjust chunk size as needed
        block_outputs = []
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i:i+chunk_size]
            print(f"Processing block {block_idx} in chunk {res_chunk_idx}-{i//chunk_size + 1} - shape: {x_chunk.shape}")  # Debugging statement
            x_chunk = block(x_chunk)
            # Save intermediate chunk
            chunk_path = os.path.join(intermediate_path, f"x_chunk_{res_chunk_idx}_{block_idx}_{i//chunk_size + 1}.pt")
            torch.save(x_chunk.cpu(), chunk_path)
            del x_chunk
            torch.cuda.empty_cache()
            gc.collect()

            # Load intermediate chunk
            x_chunk = torch.load(chunk_path)
            block_outputs.append(x_chunk)
            os.remove(chunk_path)

            # Print memory status after each chunk
            print(f"Memory after processing chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")
            print(f"Memory reserved after processing chunk {i//chunk_size + 1}: {torch.cuda.memory_reserved() / 1e9} GB")

        return torch.cat(block_outputs, dim=0).to(x.device)

if __name__ == "__main__":
    model = WarpingGenerator()
    print(model)
    batch_size = 2  # Example batch size
    R = torch.randn(batch_size, 6)  # Head pose representation (e.g., 6 DOF)
    t = torch.randn(batch_size, 50)  # Translation vectors (e.g., 50)
    z = torch.randn(batch_size, 96, 16, 28, 28)  # Appearance features (e.g., 96, 16, 28, 28)
    e = torch.randn(batch_size, 96, 16, 28, 28)  # Expression features (e.g., 96, 16, 28, 28)
    output = model(R, t, z, e)
    print(f'Output shape: {output.shape}')
