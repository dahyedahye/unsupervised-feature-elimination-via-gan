import torch
import torch.nn as nn

class CNNDiscriminator(nn.Module):
    def __init__(self, input_height, input_width):
        super(CNNDiscriminator, self).__init__()
        self.last_output_size = int((input_height / (2**5)) * (input_width / (2**5)))
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.linear = nn.Sequential(
            nn.Linear(1024*self.last_output_size, 1),
        )

    def forward(self, img):
        img = self.model(img)
        img = img.reshape(img.size(0), -1)
        validity = self.linear(img)
        return validity