import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class GazeLoss(nn.Module):
    def __init__(self):
        super(GazeLoss, self).__init__()
        self.backbone = vgg16(pretrained=True).features
        self.fc_eye = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True)
        )
        self.fc_keypoint = nn.Sequential(
            nn.Linear(2 * 68, 64),
            nn.ReLU(inplace=True)
        )
        self.gaze_head = nn.Sequential(
            nn.Linear(320, 2),
            nn.ReLU(inplace=True)
        )
        self.blink_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, eye_images, keypoints, gaze_targets, blink_targets):
        batch_size = eye_images.size(0)
        
        # Backbone forward
        features = self.backbone(eye_images)
        features = features.view(batch_size, -1)
        eye_vectors = self.fc_eye(features)
        
        # Keypoint features
        keypoints = keypoints.view(batch_size, -1)
        keypoint_vector = self.fc_keypoint(keypoints)
        
        # Gaze prediction
        gaze_input = torch.cat((eye_vectors, keypoint_vector), dim=1)
        gaze_pred = self.gaze_head(gaze_input)
        
        # Blink prediction
        blink_pred = self.blink_head(eye_vectors)
        
        # Gaze loss
        gaze_loss_mae = self.mae_loss(gaze_pred, gaze_targets)
        gaze_loss_mse = self.mse_loss(gaze_pred, gaze_targets)
        gaze_loss = 15 * gaze_loss_mae + 10 * gaze_loss_mse
        
        # Blink loss
        blink_loss = self.mse_loss(blink_pred, blink_targets)
        
        total_loss = gaze_loss + blink_loss
        return total_loss

if __name__ == "__main__":
    loss = GazeLoss()
    print(loss)
    eye_images = torch.randn(8, 3, 224, 224)  # Example input
    keypoints = torch.randn(8, 2, 68)  # Example input for keypoints (batch_size, 2, 68)
    gaze_targets = torch.randn(8, 2)  # Example input for gaze targets
    blink_targets = torch.randn(8, 1)  # Example input for blink targets
    loss_value = loss(eye_images, keypoints, gaze_targets, blink_targets)
    print(f'Loss value: {loss_value.item()}')
