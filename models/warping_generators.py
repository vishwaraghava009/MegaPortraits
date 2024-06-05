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
        print(f"ResidualBlock3D forward pass - input shape: {x.shape}")
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return F.relu(out)

class WarpingGenerator(nn.Module):
    def __init__(self):
        super(WarpingGenerator, self).__init__()
        self.initial = nn.Conv3d(248, 2048, kernel_size=1, stride=1)  
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
        device = next(self.parameters()).device

        print(f"Original shapes - R: {R.shape}, t: {t.shape}, z: {z.shape}, e: {e.shape}")
        
        batch_size = R.size(0)
        
        R = R.view(batch_size, -1, 1, 1, 1).to(device)
        t = t.view(batch_size, -1, 1, 1, 1).to(device)
        print(f"Memory after reshaping R and t: {torch.cuda.memory_allocated() / 1e9} GB")
        
        if len(z.size()) != 5 or len(e.size()) != 5:
            raise ValueError(f"Expected z and e to have 5 dimensions, got {z.size()} and {e.size()} instead.")
        
        print(f"Reshaped - R: {R.shape}, t: {t.shape}, z: {z.shape}, e: {e.shape}")

        combined_ze = torch.cat((z, e), dim=1).to(device)
        print(f"Memory after concatenating z and e: {torch.cuda.memory_allocated() / 1e9} GB")
        
        print(f"Combined z and e shape: {combined_ze.shape}")
        
        expected_channels = R.size(1) + t.size(1) + combined_ze.size(1)
        print(f"Expected channels after concatenation: {expected_channels}")
        
        combined_rt = torch.cat((R, t), dim=1).to(device)
        print(f"Memory after concatenating R and t: {torch.cuda.memory_allocated() / 1e9} GB")
        combined_rt = combined_rt.expand(-1, -1, combined_ze.size(2), combined_ze.size(3), combined_ze.size(4))
        print(f"Memory after expanding combined_rt: {torch.cuda.memory_allocated() / 1e9} GB")
        x = torch.cat((combined_rt, combined_ze), dim=1).to(device)
        print(f"Memory after final concatenation: {torch.cuda.memory_allocated() / 1e9} GB")
        
        print(f"Concatenated shape: {x.shape}")
        
        if x.size(1) != 248:
            raise ValueError(f"Concatenated tensor has {x.size(1)} channels, expected 248.")
        
        x = F.relu(self.initial(x))
        print(f"Memory after initial layer: {torch.cuda.memory_allocated() / 1e9} GB")

        print(f"Memory before res_blocks: {torch.cuda.memory_allocated() / 1e9} GB")

        x = x.cpu()
        self.res_blocks.to("cpu")
        print(f"Memory after transferring x and res_blocks to CPU: {torch.cuda.memory_allocated() / 1e9} GB")

        total_res_blocks_output = []
        for res_chunk_idx in range(0, x.size(0)):
            res_chunk = x[res_chunk_idx:res_chunk_idx + 1]  
            print(f"Processing res_chunk_idx: {res_chunk_idx}, Memory before process_res_blocks_in_chunks: {torch.cuda.memory_allocated() / 1e9} GB")
            res_chunk_output = self.process_res_blocks_in_chunks(res_chunk, res_chunk_idx)
            total_res_blocks_output.append(res_chunk_output.cpu())
            
            del res_chunk, res_chunk_output
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Memory after processing res_chunk_idx: {res_chunk_idx}: {torch.cuda.memory_allocated() / 1e9} GB")

        x = torch.cat(total_res_blocks_output, dim=0).to(device)
        self.res_blocks.to(device)
        print(f"Memory after transferring x and res_blocks back to GPU: {torch.cuda.memory_allocated() / 1e9} GB")
        
        print(f"Memory after res_blocks: {torch.cuda.memory_allocated() / 1e9} GB")

        x = self.final(x)
        
        print(f"Memory after final layer: {torch.cuda.memory_allocated() / 1e9} GB")
        
        return self.tanh(x)

    def process_res_blocks_in_chunks(self, x, res_chunk_idx):
        """
        Process the res_blocks in chunks to manage memory usage better.
        """
        chunk_size = 1  
        res_block_outputs = []
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i:i+chunk_size]
            print(f"Memory before block processing - chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")
            for block_idx, block in enumerate(self.res_blocks):
                if block_idx >= 8:  
                    x_chunk = self.process_block_in_smaller_units(block, x_chunk, res_chunk_idx, block_idx)
                else:
                    x_chunk = self.process_block_in_chunks(block, x_chunk, res_chunk_idx, block_idx)

            chunk_path = os.path.join(intermediate_path, f"res_chunk_{res_chunk_idx}_{i//chunk_size + 1}.pt")
            torch.save(x_chunk.cpu(), chunk_path)
            print(f"Memory after saving res_chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

            del x_chunk
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Memory after deleting res_chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

            x_chunk = torch.load(chunk_path).to(x.device)
            res_block_outputs.append(x_chunk)
            os.remove(chunk_path)
            print(f"Memory after loading res_chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

            print(f"Memory after processing res_chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

        return torch.cat(res_block_outputs, dim=0).to(x.device)

    def process_block_in_chunks(self, block, x, res_chunk_idx, block_idx):
        """
        Process each block within res_blocks in chunks.
        """
        chunk_size = 1  
        block_outputs = []
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i:i+chunk_size]
            print(f"Processing block {block_idx} in chunk {res_chunk_idx}-{i//chunk_size + 1} - shape: {x_chunk.shape}")  
            print(f"Memory before block forward pass - chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

            try:
                x_chunk = block(x_chunk)
            except Exception as e:
                print(f"Error during block forward pass: {e}")
                raise e

            print(f"Memory after block forward pass - chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

            chunk_path = os.path.join(intermediate_path, f"x_chunk_{res_chunk_idx}_{block_idx}_{i//chunk_size + 1}.pt")
            torch.save(x_chunk.cpu(), chunk_path)
            print(f"Memory after saving chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")
            del x_chunk
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Memory after deleting chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

            x_chunk = torch.load(chunk_path)
            block_outputs.append(x_chunk)
            os.remove(chunk_path)
            print(f"Memory after loading chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")

            print(f"Memory after processing chunk {i//chunk_size + 1}: {torch.cuda.memory_allocated() / 1e9} GB")
            print(f"Memory reserved after processing chunk {i//chunk_size + 1}: {torch.cuda.memory_reserved() / 1e9} GB")

        return torch.cat(block_outputs, dim=0).to(x.device)

    def process_block_in_smaller_units(self, block, x, res_chunk_idx, block_idx):
        """
        Special handling for blocks 8 and greater to process in even smaller units.
        """
        unit_size = 64  
        partial_output = None  
        depth_splits = (x.size(2) + unit_size - 1) // unit_size
        height_splits = (x.size(3) + unit_size - 1) // unit_size
        width_splits = (x.size(4) + unit_size - 1) // unit_size

        for d in range(depth_splits):
            for h in range(height_splits):
                for w in range(width_splits):
                    start_d = d * unit_size
                    end_d = min(start_d + unit_size, x.size(2))
                    start_h = h * unit_size
                    end_h = min(start_h + unit_size, x.size(3))
                    start_w = w * unit_size
                    end_w = min(start_w + unit_size, x.size(4))

                    x_unit = x[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
                    print(f"Processing block {block_idx} unit ({start_d}, {start_h}, {start_w}) - shape: {x_unit.shape}")  
                    print(f"Memory before block forward pass - unit ({start_d}, {start_h}, {start_w}): {torch.cuda.memory_allocated() / 1e9} GB")

                    try:
                        x_unit = block(x_unit)
                    except Exception as e:
                        print(f"Error during block forward pass: {e}")
                        raise e

                    print(f"Memory after block forward pass - unit ({start_d}, {start_h}, {start_w}): {torch.cuda.memory_allocated() / 1e9} GB")

                    # concatenating with existing partial output
                    if partial_output is None:
                        partial_output = x_unit
                    else:
                        partial_output = torch.cat((partial_output, x_unit), dim=2).to(x.device)

                    partial_output_path = os.path.join(intermediate_path, f"partial_output_{res_chunk_idx}_{block_idx}.pt")
                    torch.save(partial_output.cpu(), partial_output_path)
                    print(f"Memory after saving partial output: {torch.cuda.memory_allocated() / 1e9} GB")

                    del x_unit
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"Memory after deleting unit ({start_d}, {start_h}, {start_w}): {torch.cuda.memory_allocated() / 1e9} GB")

                    partial_output = torch.load(partial_output_path).to(x.device)
                    os.remove(partial_output_path)
                    print(f"Memory after loading partial output: {torch.cuda.memory_allocated() / 1e9} GB")

                    print(f"Memory after processing unit ({start_d}, {start_h}, {start_w}): {torch.cuda.memory_allocated() / 1e9} GB")
                    print(f"Memory reserved after processing unit ({start_d}, {start_h}, {start_w}): {torch.cuda.memory_reserved() / 1e9} GB")

        return partial_output.to(x.device)


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
