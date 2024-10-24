from PIL import Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlignedDataset(Dataset):
    def __init__(
            self, 
            input_dir,
            target_dir,
            ext="jpg",
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)

        self.input_files = sorted(self.input_dir.glob("*."+ext))
        self.target_files = sorted(self.target_dir.glob("*."+ext))
        
        assert len(self.input_files) == len(self.target_files), "Number of input files and target files must be the same"
        
        self.ext = ext
        self.setup_transform()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]

        if self.ext in ["jpg", "jpeg", "png"]:
            # [0, 255]
            # [H, W, C]
            # uint8
            input_image = np.array(Image.open(input_file))   # [0, 255]
            target_image = np.array(Image.open(target_file)) # [0, 255]

            transformed = self.transform(image=input_image, image_target=target_image)
            input_image = transformed["image"]
            target_image = transformed["image_target"]
        else:
            raise NotImplementedError("File extension not supported")
        
        return input_image, target_image
    
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
        else:
            raise NotImplementedError("File extension not supported")

    def denormalize(self, x):
        if self.ext in ["jpg", "jpeg", "png"]:
            MaxVal = self.MaxVal
            MinVal = self.MinVal
            m = self.m
            s = self.s

            x = x * s + m
            x = np.clip(x, MinVal, MaxVal)
            x = x.astype(np.uint8)
            return x
        else:
            raise NotImplementedError("File extension not supported")

    def save_image(self, x, filename):
        # x: [C, H, W], [-1, 1], torch.float32
        x = x.detach().cpu().numpy()
        x = self.denormalize(x)  
        x = np.transpose(x, (1, 2, 0))

        if self.ext in ["jpg", "jpeg", "png"]:
            Image.fromarray(x).save(filename)
        # plt.figure()
        # plt.subplot(1, 1, 1)
        # plt.imshow(x, aspect="equal")
        # plt.axis("off")
        # plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        # plt.close()

    def create_figure(self, real_input, real_target, fake_target):
        if self.ext in ["jpg", "jpeg", "png"]:
            real_input = real_input.detach().cpu().numpy()
            real_target = real_target.detach().cpu().numpy()
            fake_target = fake_target.detach().cpu().numpy()

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
    
if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    with open("configs/default.yaml") as file:
        cfg = yaml.safe_load(file)

    dataset = AlignedDataset(
        input_dir = cfg['data']['train_input_dir'],
        target_dir = cfg['data']['train_target_dir'],
        ext = cfg['data']['ext']
    )
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        dataset.save_image(x[0], "x_input.png")
        dataset.save_image(y[0], "y_target.png")
        import sys
        sys.exit()