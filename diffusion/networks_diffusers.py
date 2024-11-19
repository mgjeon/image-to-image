from torch import nn
from diffusers import UNet2DModel

class DiffusersUNet2DModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = UNet2DModel(*args, **kwargs)
    
    def forward(self, x, t):
        return self.model(x, t).sample
