import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.iresnet import iresnet100  # Import the iresnet100 model

class CosineSimilarityLoss(nn.Module):
    def __init__(self, checkpoint_path):
        super(CosineSimilarityLoss, self).__init__()
        self.face_model = iresnet100(pretrained=False)
        self.face_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.face_model.eval()  # Set the model to evaluation mode

    def forward(self, I_s, I_d):
        # Extract deep face identity features
        with torch.no_grad():
            features_s = self.face_model(I_s)
            features_d = self.face_model(I_d)
        
        # Compute the cosine similarity loss between identity features
        identity_loss = -F.cosine_similarity(features_s, features_d).mean()
        return identity_loss

if __name__ == "__main__":
    checkpoint_path = 'checkpoints/backbone.pth'  # Path to the pre-trained checkpoint
    loss = CosineSimilarityLoss(checkpoint_path)
    print(loss)
    
    # Example input tensors
    I_s = torch.randn(1, 3, 112, 112)  # Example input size for iresnet100
    I_d = torch.randn(1, 3, 112, 112)

    loss_value = loss(I_s, I_d)
    print(f'Loss value: {loss_value.item()}')
