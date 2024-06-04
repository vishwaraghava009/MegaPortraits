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
from models.conv3d import Conv3D
from models.conv2d import Conv2D
from models.discriminator import PatchGANDiscriminator
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import AdversarialLoss
from losses.cycle_consistency_loss import CycleConsistencyLoss
from losses.gaze_loss import GazeLoss

intermediate_path = "/content/drive/MyDrive/VASA-1-master/intermediate_chunks"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_intermediate_chunks(chunk, index):
    os.makedirs(intermediate_path, exist_ok=True)
    torch.save(chunk, os.path.join(intermediate_path, f"chunk_{index}.pt"))

def load_intermediate_chunks(index):
    return torch.load(os.path.join(intermediate_path, f"chunk_{index}.pt"))

def delete_intermediate_chunks(index):
    os.remove(os.path.join(intermediate_path, f"chunk_{index}.pt"))

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
    
    # Debug: Check the number of samples in the dataset
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
            chunk_size = 2  # Adjust chunk size as needed
            num_chunks = (source_image.size(0) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, source_image.size(0))

                source_chunk = source_image[start:end]
                driving_chunk = driving_frame[start:end]

                optimizer_G.zero_grad()
                optimizer_D.zero_grad()

                # Forward pass through the models
                appearance_features = appearance_encoder(source_chunk)
                driving_features = appearance_encoder(driving_chunk)
                head_pose, expression = motion_encoder(driving_chunk)

                print(f'Original shapes - R: {head_pose.shape}, t: {expression.shape}, z: {appearance_features.shape}, e: {driving_features.shape}')
                head_pose = head_pose.view(head_pose.size(0), -1, 1, 1, 1)
                expression = expression.view(expression.size(0), -1, 1, 1, 1)
                print(f'Reshaped - R: {head_pose.shape}, t: {expression.shape}, z: {appearance_features.shape}, e: {driving_features.shape}')
                
                combined_ze = torch.cat((appearance_features, driving_features), dim=1)
                print(f'Combined z and e shape: {combined_ze.shape}')
                expected_channels = head_pose.size(1) + expression.size(1) + combined_ze.size(1)
                print(f'Expected channels after concatenation: {expected_channels}')
                
                warp_chunks = []
                for wc_idx in range(num_chunks):
                    warp_chunk = warping_generator(head_pose.detach(), expression.detach(), appearance_features.detach(), driving_features.detach())
                    warp_chunks.append(warp_chunk.cpu())
                    del warp_chunk
                    torch.cuda.empty_cache()
                    gc.collect()

                warped_features = torch.cat(warp_chunks, dim=0).to(device)
                del warp_chunks
                torch.cuda.empty_cache()
                gc.collect()

                canonical_volume = conv3d(warped_features)
                generated_image = conv2d(canonical_volume)

                # Compute losses
                perc_loss = perceptual_loss(generated_image, driving_chunk)
                adv_loss = adversarial_loss(discriminator(driving_chunk), discriminator(generated_image))
                cycle_loss = cycle_consistency_loss(source_chunk, driving_chunk)
                gaze_loss_value = gaze_loss(source_chunk, driving_chunk, head_pose, expression)

                total_loss = perc_loss + adv_loss + cycle_loss + gaze_loss_value
                total_loss.backward()
                optimizer_G.step()
                optimizer_D.step()

                if i % log_interval == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Chunk [{chunk_idx}/{num_chunks}], Loss: {total_loss.item()}")

                # Force delete intermediate tensors
                del source_chunk, driving_chunk, appearance_features, driving_features, head_pose, expression, warped_features, canonical_volume, generated_image
                torch.cuda.empty_cache()
                gc.collect()

        # Save checkpoints
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': appearance_encoder.state_dict(),
                'optimizer': optimizer_G.state_dict()
            }, checkpoint_path, f'checkpoint_{epoch+1}.pth')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': discriminator.state_dict(),
                'optimizer': optimizer_D.state_dict()
            }, checkpoint_path, f'checkpoint_D_{epoch+1}.pth')

    # Save latest checkpoints
    save_checkpoint({
        'epoch': epoch,
        'state_dict': appearance_encoder.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }, checkpoint_path, 'latest_checkpoint.pth')
    save_checkpoint({
        'epoch': epoch,
        'state_dict': discriminator.state_dict(),
        'optimizer': optimizer_D.state_dict()
    }, checkpoint_path, 'latest_checkpoint_D.pth')

if __name__ == "__main__":
    with open('configs/base_model.yaml', 'r') as file:
        config = yaml.safe_load(file)
    train(config)
