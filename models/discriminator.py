import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (input_channels) x 256 x 256
            spectral_norm(nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 31 x 31
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))
            # state size. 1 x 30 x 30
        )

    def forward(self, input):
        return self.model(input)

if __name__ == "__main__":
    model = PatchGANDiscriminator()
    print(model)
    test_input = torch.randn(1, 3, 256, 256)
    test_output = model(test_input)
    print(f'Output shape: {test_output.shape}')
