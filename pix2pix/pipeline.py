from PIL import Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# AlignedDataset ========================================================================================
class AlignedDataset(Dataset):
    def __init__(
            self, 
            input_dir,
            target_dir,
            image_size=256,
            ext="jpg",
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)

        self.input_files = sorted(self.input_dir.glob("*."+ext))
        self.target_files = sorted(self.target_dir.glob("*."+ext))
        
        assert len(self.input_files) == len(self.target_files), "Number of input files and target files must be the same"
        
        self.ext = ext
        self.image_size = image_size
        self.setup_transform()

    def __len__(self):
        return len(self.input_files)

# Get item ===============================================================================================
    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]

        if self.ext in ["jpg", "jpeg", "png"]:
            # [0, 255]
            # [H, W, C]
            # np.uint8
            input_image = np.array(Image.open(input_file))   # [0, 255]
            target_image = np.array(Image.open(target_file)) # [0, 255]

            transformed = self.transform(image=input_image, image_target=target_image)
            # [C, H, W] torch.float32
            input_image = transformed["image"]
            target_image = transformed["image_target"]

        elif self.ext in ["npy"]:
            # [-1, 1]
            # [C, H, W]
            # np.float32
            # [C, H, W] -> [H, W, C] for albumentations
            input_image = np.load(input_file).astype(np.float32).transpose(1, 2, 0) 
            target_image = np.load(target_file).astype(np.float32).transpose(1, 2, 0) 

            transformed = self.transform(image=input_image, image_target=target_image)
            # [C, H, W] torch.float32
            input_image = transformed["image"]
            target_image = transformed["image_target"]
        else:
            raise NotImplementedError("File extension not supported")
        
        return input_image, target_image
    
# Setup transform ========================================================================================
    def setup_transform(self):
        if self.ext in ["jpg", "jpeg", "png"]:
            # [0, 255]
            # [H, W, C]
            # uint8
            MinVal = 0
            MaxVal = 255
            m = (MinVal + MaxVal) / 2
            s = (MaxVal - MinVal) / 2
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                # A.VerticalFlip(p=0.5),
                A.Normalize(mean=[m, m, m], std=[s, s, s], max_pixel_value=1.0),  # [-1, 1],
                ToTensorV2(),
            ], 
                additional_targets = {
                    'image_target': 'image',
                }
            )
            self.MaxVal = MaxVal
            self.MinVal = MinVal
            self.m = m
            self.s = s
        elif self.ext in ["npy"]:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                # A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ], 
                additional_targets = {
                    'image_target': 'image',
                }
            )
        else:
            raise NotImplementedError("File extension not supported")

# Denormalize to [0, 255] ===============================================================================
    def denormalize(self, x):
        if self.ext in ["jpg", "jpeg", "png"]:
            MaxVal = self.MaxVal
            MinVal = self.MinVal
            m = self.m
            s = self.s

            x = x * s + m
            x = np.clip(x, MinVal, MaxVal)
            x = x.astype(np.uint8)  # [0, 255]
            return x
        elif self.ext in ["npy"]:
            MinVal = 0
            MaxVal = 255
            m = (MinVal + MaxVal) / 2
            s = (MaxVal - MinVal) / 2

            x = x * s + m
            x = np.clip(x, MinVal, MaxVal)
            x = x.astype(np.uint8)  # [0, 255]
            return x
        else:
            raise NotImplementedError("File extension not supported")

# Save image =============================================================================================
    def save_image(self, x, filename):
        # x: [C, H, W], [-1, 1], torch.float32
        assert x.shape[0] == 3 or x.shape[0] == 1, "Only support RGB or Grayscale image"
        x = x.detach().cpu().numpy()
        x = self.denormalize(x)        # [0, 255]
        x = np.transpose(x, (1, 2, 0)) # [C, H, W] -> [H, W, C]
        if self.ext in ["jpg", "jpeg", "png"]:
            Image.fromarray(x).save(filename)
        elif self.ext in ["npy"]:
            if x.shape[2] == 1:
                x = np.squeeze(x, axis=2)
            Image.fromarray(x).save(filename)
        else:
            raise NotImplementedError("File extension not supported")

# Create figure ==========================================================================================
    def create_figure(self, real_input, real_target, fake_target):
        # [C, H, W], [-1, 1], torch.float32
        real_input = real_input.detach().cpu().numpy()
        real_target = real_target.detach().cpu().numpy()
        fake_target = fake_target.detach().cpu().numpy()

        if self.ext in ["jpg", "jpeg", "png"]:
            real_input = self.denormalize(real_input)
            real_target = self.denormalize(real_target)
            fake_target = self.denormalize(fake_target)

            real_input = np.transpose(real_input, (1, 2, 0))
            real_target = np.transpose(real_target, (1, 2, 0))
            fake_target = np.transpose(fake_target, (1, 2, 0))
            
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(real_input, aspect="equal")
            ax.axis("off")
            ax.set_title("Input")
            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(real_target, aspect="equal")
            ax.axis("off")
            ax.set_title("Target")
            ax = fig.add_subplot(1, 3, 3)
            ax.imshow(fake_target, aspect="equal")
            ax.axis("off")
            ax.set_title("Generated")
            fig.tight_layout()
            return fig
        
        elif self.ext in ["npy"]:
            in_channels = real_input.shape[0]
            out_channels_real = real_target.shape[0]
            out_channels_fake = fake_target.shape[0]

            cols = in_channels + out_channels_real + out_channels_fake
            height = 3
            width = (cols + 1)*height

            fig, axs = plt.subplots(1, cols, figsize=(width, height))
            axs = axs.flatten()
            for i in range(in_channels):
                axs[i].imshow(real_input[i], cmap="gray", vmin=-1, vmax=1, aspect="equal", origin="lower")
                axs[i].axis("off")
                axs[i].set_title(f"Input {i}")
            for i in range(out_channels_real):
                axs[in_channels+i].imshow(real_target[i], cmap="gray", vmin=-1, vmax=1, aspect="equal", origin="lower")
                axs[in_channels+i].axis("off")
                axs[in_channels+i].set_title(f"Target {i}")
            for i in range(out_channels_fake):
                axs[in_channels+out_channels_real+i].imshow(fake_target[i], cmap="gray", vmin=-1, vmax=1, aspect="equal", origin="lower")
                axs[in_channels+out_channels_real+i].axis("off")
                axs[in_channels+out_channels_real+i].set_title(f"Generated {i}")
                
            fig.tight_layout()
            return fig
        else:
            raise NotImplementedError("File extension not supported")
          

# Test =================================================================================================
if __name__ == "__main__":
    import yaml
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    dataset = AlignedDataset(
        input_dir = cfg['data']['train']['input_dir'],
        target_dir = cfg['data']['train']['target_dir'],
        ext = cfg['data']['ext']
    )
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        # dataset.save_image(x[0], "x_input.png")
        # dataset.save_image(y[0], "y_target.png")
        fig = dataset.create_figure(x[0], y[0], y[0])
        fig.savefig("figure.png")
        import sys
        sys.exit()