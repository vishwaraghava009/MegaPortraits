import os
import sys
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset import MegaPortraitDataset
from models.appearance_encoder import AppearanceEncoder
from models.motion_encoder import MotionEncoder
from models.warping_generators import WarpingGenerator
from models.conv3d import Conv3D, ResidualBlock3D
from models.conv2d import Conv2D
from models.discriminator import PatchGANDiscriminator
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import AdversarialLoss
from losses.cycle_consistency_loss import CycleConsistencyLoss
from losses.gaze_loss import GazeLoss

intermediate_path = "/content/drive/MyDrive/VASA-1-master/intermediate_chunks"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_intermediate_chunks(chunk, index, prefix):
    os.makedirs(intermediate_path, exist_ok=True)
    torch.save(chunk, os.path.join(intermediate_path, f"{prefix}_chunk_{index}.pt"))

def load_intermediate_chunks(index, prefix):
    return torch.load(os.path.join(intermediate_path, f"{prefix}_chunk_{index}.pt"))

def save_checkpoint(state, checkpoint_path, filename):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_path, filename))

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded successfully from '{checkpoint_path}'")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")

def train(config):
    # Hyperparameters
    data_path = config['data_path']
    checkpoint_path = config['checkpoint_path']
    log_path = config['log_path']
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['lr']
    log_interval = config['log_interval']
    checkpoint_interval = config['checkpoint_interval']
    resume = config['resume']

    # Data loading and transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = MegaPortraitDataset(data_path=data_path, transform=transform)
    
    print(f"Number of samples in the dataset: {len(dataset)}")
    
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data path and ensure the dataset contains valid samples.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model initialization
    appearance_encoder = AppearanceEncoder().to(device)
    motion_encoder = MotionEncoder().to(device)
    warping_generator = WarpingGenerator().to(device)
    conv3d = Conv3D().to(device)
    conv2d = Conv2D().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # Losses
    perceptual_loss = PerceptualLoss().to(device)
    adversarial_loss = AdversarialLoss().to(device)
    cycle_consistency_loss = CycleConsistencyLoss().to(device)
    gaze_loss = GazeLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(list(appearance_encoder.parameters()) + 
                             list(motion_encoder.parameters()) + 
                             list(warping_generator.parameters()) + 
                             list(conv3d.parameters()) + 
                             list(conv2d.parameters()), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Resume from checkpoint if needed
    start_epoch = 0
    if resume:
        load_checkpoint(os.path.join(checkpoint_path, 'latest_checkpoint.pth'), appearance_encoder, optimizer_G)
        load_checkpoint(os.path.join(checkpoint_path, 'latest_checkpoint_D.pth'), discriminator, optimizer_D)
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, num_epochs):
        for i, (source_image, driving_frame) in enumerate(dataloader):
            source_image = source_image.to(device)
            driving_frame = driving_frame.to(device)

            # Split into chunks to avoid out-of-memory errors
            chunk_size = 1  
            num_chunks = (source_image.size(0) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, source_image.size(0))

                source_chunk = source_image[start:end]
                driving_chunk = driving_frame[start:end]

                optimizer_G.zero_grad()
                optimizer_D.zero_grad()

                # Forward pass through the models
                print(f"Memory before appearance encoder: {torch.cuda.memory_allocated() / 1e9} GB")
                appearance_features = appearance_encoder(source_chunk)
                driving_features = appearance_encoder(driving_chunk)
                print(f"Memory after appearance encoder: {torch.cuda.memory_allocated() / 1e9} GB")

                print(f"Memory before motion encoder: {torch.cuda.memory_allocated() / 1e9} GB")
                head_pose, expression = motion_encoder(driving_chunk)
                print(f"Memory after motion encoder: {torch.cuda.memory_allocated() / 1e9} GB")

                print(f"Memory before warping generator: {torch.cuda.memory_allocated() / 1e9} GB")
                warp_chunk = warping_generator(head_pose.detach(), expression.detach(), appearance_features.detach(), driving_features.detach())
                print(f"Memory after warping generator: {torch.cuda.memory_allocated() / 1e9} GB")
                print(f"Warp chunk shape: {warp_chunk.shape}")  # Debugging statement

                save_intermediate_chunks(warp_chunk, chunk_idx, "warp_chunk")
                del warp_chunk, appearance_features, driving_features, head_pose, expression
                torch.cuda.empty_cache()
                gc.collect()

                warp_chunk = load_intermediate_chunks(chunk_idx, "warp_chunk").to(device)
                
                print(f"Memory before conv3d: {torch.cuda.memory_allocated() / 1e9} GB")
                # Split conv3d forward pass into smaller parts to debug memory usage
                x = warp_chunk
                for j, layer in enumerate(conv3d.res_blocks):
                    print(f"Before layer {j} ({layer.__class__.__name__}): shape = {x.shape}, memory = {torch.cuda.memory_allocated() / 1e9} GB")
                    x = layer(x, j) if isinstance(layer, ResidualBlock3D) else layer(x)
                    save_intermediate_chunks(x, j, f"conv3d_layer_{chunk_idx}")
                    del x
                    torch.cuda.empty_cache()
                    x = load_intermediate_chunks(j, f"conv3d_layer_{chunk_idx}")
                    print(f"After layer {j} ({layer.__class__.__name__}): shape = {x.shape}, memory = {torch.cuda.memory_allocated() / 1e9} GB")
                    if torch.cuda.memory_allocated() / 1e9 > 10:
                        print(f"Memory usage exceeded limit after layer {j}")
                        break
                canonical_volume = x

                print(f"Memory before conv2d: {torch.cuda.memory_allocated() / 1e9} GB")
                generated_image = conv2d(canonical_volume)
                print(f"Memory after conv2d: {torch.cuda.memory_allocated() / 1e9} GB")

                # Compute losses
                print(f"Memory before loss computation: {torch.cuda.memory_allocated() / 1e9} GB")
                perc_loss = perceptual_loss(generated_image, driving_chunk)
                adv_loss = adversarial_loss(discriminator, driving_chunk, generated_image)
                cycle_loss = cycle_consistency_loss(source_chunk, driving_chunk)
                gaze_loss_value = gaze_loss(source_chunk, driving_chunk, head_pose, expression)

                total_loss = perc_loss + adv_loss + cycle_loss + gaze_loss_value
                print(f"Memory after loss computation: {torch.cuda.memory_allocated() / 1e9} GB")

                print(f"Memory before backward: {torch.cuda.memory_allocated() / 1e9} GB")
                total_loss.backward()
                print(f"Memory after backward: {torch.cuda.memory_allocated() / 1e9} GB")
                optimizer_G.step()
                optimizer_D.step()

                # Save intermediate results
                save_intermediate_chunks(generated_image, i * num_chunks + chunk_idx, "generated_image")

                del source_chunk, driving_chunk, warp_chunk, canonical_volume, generated_image
                torch.cuda.empty_cache()
                gc.collect()  

                if i % log_interval == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Chunk [{chunk_idx}/{num_chunks}], Loss: {total_loss.item()}")

    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': appearance_encoder.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }, checkpoint_path, 'final_checkpoint_appearance_encoder.pth')
    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': motion_encoder.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }, checkpoint_path, 'final_checkpoint_motion_encoder.pth')
    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': warping_generator.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }, checkpoint_path, 'final_checkpoint_warping_generator.pth')
    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': conv3d.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }, checkpoint_path, 'final_checkpoint_conv3d.pth')
    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': conv2d.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }, checkpoint_path, 'final_checkpoint_conv2d.pth')
    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': discriminator.state_dict(),
        'optimizer': optimizer_D.state_dict()
    }, checkpoint_path, 'final_checkpoint_discriminator.pth')

if __name__ == "__main__":
    with open('configs/base_model.yaml', 'r') as file:
        config = yaml.safe_load(file)
    train(config)