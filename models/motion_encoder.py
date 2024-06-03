import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        # Head Pose Prediction Network (pre-trained)
        self.head_pose = resnet18(pretrained=True)
        self.head_pose.fc = nn.Linear(self.head_pose.fc.in_features, 6)  # Output for rotations and translations

        # Expression Prediction Network (trained from scratch)
        self.expression = resnet18(pretrained=False)
        self.expression.fc = nn.Linear(self.expression.fc.in_features, 50)  # Latent expression descriptors

    def forward(self, x):
        pose = self.head_pose(x)
        expr = self.expression(x)
        return pose, expr

if __name__ == "__main__":
    model = MotionEncoder()
    print(model)
    test_input = torch.randn(1, 3, 224, 224)
    pose_output, expr_output = model(test_input)
    print(f'Pose output shape: {pose_output.shape}')
    print(f'Expression output shape: {expr_output.shape}')
