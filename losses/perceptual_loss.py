import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, vgg16
from facenet_pytorch import InceptionResnetV1

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_weights='imagenet'):
        super(PerceptualLoss, self).__init__()
        
        # ImageNet Loss (VGG19)
        self.vgg19 = vgg19(pretrained=True).features[:36]  # Up to conv5_4
        for param in self.vgg19.parameters():
            param.requires_grad = False
        
        # Face Loss (VGGFace / InceptionResnetV1)
        self.face_model = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.face_model.parameters():
            param.requires_grad = False
        
        # Gaze Loss (VGG16)
        self.vgg16 = vgg16(pretrained=True).features[:23]  # Up to conv4_3
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # VGG19 (ImageNet Loss)
        x_vgg19 = self.vgg19(x)
        y_vgg19 = self.vgg19(y)
        loss_imagenet = F.l1_loss(x_vgg19, y_vgg19)
        
        # VGGFace / InceptionResnetV1 (Face Loss)
        x_face = self.face_model(x)
        y_face = self.face_model(y)
        loss_face = F.l1_loss(x_face, y_face)
        
        # VGG16 (Gaze Loss)
        x_vgg16 = self.vgg16(x)
        y_vgg16 = self.vgg16(y)
        loss_gaze = F.l1_loss(x_vgg16, y_vgg16)

        # Combine losses
        total_loss = loss_imagenet + loss_face + loss_gaze
        return total_loss

if __name__ == "__main__":
    loss = PerceptualLoss()
    print(loss)
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 224, 224)
    loss_value = loss(x, y)
    print(f'Loss value: {loss_value.item()}')
