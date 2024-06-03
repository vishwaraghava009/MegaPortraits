import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.iresnet import iresnet100  # Import the iresnet100 model

class PairwiseHeadPoseFacialDynamicsLoss(nn.Module):
    def __init__(self, checkpoint_path):
        super(PairwiseHeadPoseFacialDynamicsLoss, self).__init__()
        self.face_model = iresnet100(pretrained=False)
        self.face_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.face_model.eval()  # Set the model to evaluation mode
        self.l1_loss = nn.L1Loss()

    def forward(self, I_s, I_t, z_pose_s, z_dyn_t, z_pose_t):
        # Extract deep face identity features
        with torch.no_grad():
            features_s = self.face_model(I_s)
            features_t = self.face_model(I_t)
        
        # Transfer head pose
        combined_s = torch.cat([features_s, z_pose_s, z_dyn_t], dim=1)
        combined_t = torch.cat([features_t, z_pose_t, z_dyn_t], dim=1)
        
        # Compute the L1 loss between the combined features
        l1_loss = self.l1_loss(combined_s, combined_t)

        return l1_loss

if __name__ == "__main__":
    checkpoint_path = 'checkpoints/backbone.pth'  # Path to the pre-trained checkpoint
    loss = PairwiseHeadPoseFacialDynamicsLoss(checkpoint_path)
    print(loss)
    
    # Example input tensors
    I_s = torch.randn(1, 3, 112, 112)  # Example input size for iresnet100
    I_t = torch.randn(1, 3, 112, 112)
    z_pose_s = torch.randn(1, 6)  # Example size for z_pose of source
    z_dyn_t = torch.randn(1, 50)  # Example size for z_dyn of target
    z_pose_t = torch.randn(1, 6)  # Example size for z_pose of target

    loss_value = loss(I_s, I_t, z_pose_s, z_dyn_t, z_pose_t)
    print(f'Loss value: {loss_value.item()}')
