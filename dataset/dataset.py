import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2

class MegaPortraitDataset(Dataset):
    def __init__(self, data_path, transform=None, max_frames=60):
        self.data_path = data_path
        self.transform = transform
        self.video_files = self.load_video_files()
        self.max_frames = max_frames  # Maximum number of frames to pad/truncate to

    def load_video_files(self):
        video_files = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
        return video_files

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = self.load_video_frames(video_path)
        
        if len(frames) < 2:
            raise ValueError(f"Video at {video_path} does not have enough frames.")

        source_frame, driving_frame = random.sample(frames, 2)

        if self.transform:
            source_frame = self.transform(source_frame)
            driving_frame = self.transform(driving_frame)

        return source_frame, driving_frame

    def load_video_frames(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        video_capture.release()
        return frames

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = MegaPortraitDataset(data_path='/content/drive/MyDrive/VASA-1-master/video', transform=transform)
    print(f'Dataset size: {len(dataset)}')
    sample = dataset[0]
    print(f'Sample shapes: {sample[0].shape}, {sample[1].shape}')
