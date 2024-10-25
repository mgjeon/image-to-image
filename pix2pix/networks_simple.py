# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix

import torch
from torch import nn


# Block =============================================================================
class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


# Generator =============================================================================
class SimpleGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = GeneratorBlock(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = GeneratorBlock(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = GeneratorBlock(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = GeneratorBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = GeneratorBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = GeneratorBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = GeneratorBlock(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = GeneratorBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = GeneratorBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = GeneratorBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = GeneratorBlock(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = GeneratorBlock(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = GeneratorBlock(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))
    

# Discriminator =============================================================================
class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                DiscriminatorBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return x
    
# Test =============================================================================
if __name__ == "__main__":
    batch_size = 1
    in_channels = 2
    out_channels = 1
    height = 1024
    width = 1024

    real_input  = torch.randn((batch_size,  in_channels, height, width))  # Input
    real_target = torch.randn((batch_size, out_channels, height, width))  # Target
    print("Input")
    print(real_input.shape)
    print("Target")
    print(real_target.shape)

    gen = SimpleGenerator(in_channels=in_channels, out_channels=out_channels)
    fake_target = gen(real_input)
    print("Generator")
    print(fake_target.shape)
    assert fake_target.shape == real_target.shape

    disc = SimpleDiscriminator(in_channels=in_channels + out_channels)
    preds = disc(torch.cat([real_input, real_target], dim=1))
    print("Discriminator")
    print(preds.shape)