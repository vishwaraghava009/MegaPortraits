import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.net(x)

class AdversarialLoss(nn.Module):
    def __init__(self, feature_matching_weight=10):
        super(AdversarialLoss, self).__init__()
        self.feature_matching_weight = feature_matching_weight
        self.adv_loss = nn.ReLU()
        self.feature_loss = nn.L1Loss()

    def forward(self, discriminator, real, fake):
        real_preds = discriminator(real)
        fake_preds = discriminator(fake)

        real_loss = self.adv_loss(1.0 - real_preds).mean()
        fake_loss = self.adv_loss(1.0 + fake_preds).mean()
        adv_loss = (real_loss + fake_loss) * 0.5

        # Feature-matching loss
        real_features = real_preds[:-1]
        fake_features = fake_preds[:-1]
        fm_loss = sum([self.feature_loss(real_feat, fake_feat.detach()) for real_feat, fake_feat in zip(real_features, fake_features)])
        total_loss = adv_loss + self.feature_matching_weight * fm_loss

        return total_loss

if __name__ == "__main__":
    discriminator = PatchGANDiscriminator()
    loss = AdversarialLoss()
    print(loss)
    real = torch.randn(1, 3, 256, 256)
    fake = torch.randn(1, 3, 256, 256)
    loss_value = loss(discriminator, real, fake)
    print(f'Loss value: {loss_value.item()}')
