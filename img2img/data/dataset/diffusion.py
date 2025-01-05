import torch
from torch.utils.data import Dataset
from .pipeline import AlignedDataset
from img2img.utils.diffusion.noise_schedule import get_beta_schedule

class DiffusionAlignedDataset(Dataset):
    def __init__(self, cfg, *args, **kwargs):
        self.dataset = AlignedDataset(*args, **kwargs)
        self.cfg = cfg
        self.num_timesteps = cfg['params']['diffusion']['num_timesteps']
        betas = get_beta_schedule(
            beta_schedule=cfg['params']['diffusion']['beta_schedule'],
            beta_start=cfg['params']['diffusion']['beta_start'],
            beta_end=cfg['params']['diffusion']['beta_end'],
            num_diffusion_timesteps=self.num_timesteps
        )
        betas = torch.from_numpy(betas).float()
        self.alphas = (1-betas).cumprod(dim=0)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_image, target_image, input_name, target_name = self.dataset[idx]

# Generate Noise ===============================================================
        # (ch_out, h, w)
        e = torch.randn_like(target_image)
        if self.cfg['params']['diffusion']['clip_noise']:
            e = e.clamp(-1, 1)
        n = e.shape[0]

        num_timesteps = self.cfg['params']['diffusion']['num_timesteps']

# Antithetic Sampling for Diffusion Timesteps ==================================
        # (1,)
        if self.cfg['params']['diffusion']['fast_ddpm']:
            skip = num_timesteps // self.cfg['params']['sampling']['timesteps']
            t_intervals = torch.arange(-1, num_timesteps, skip)
            t_intervals[0] = 0

            idx_1 = torch.randint(low=0, high=len(t_intervals), size=(n//2 + 1,))
            idx_2 = len(t_intervals)-idx_1-1
            idx = torch.cat([idx_1, idx_2], dim=0)[:n]
            t = t_intervals[idx]
        else:
            t = torch.randint(
                low=0, high=num_timesteps, size=(n//2 + 1,)
            )
            t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]

# Add Noise to Real Target =====================================================
        # (ch_out, h, w) x (1, 1, 1)
        a = self.alphas.index_select(dim=0, index=t).view(1, 1, 1)
        xt = target_image*a.sqrt() + e*(1.0 - a).sqrt()

# Concatenate Inputs ===========================================================
        return t, e, xt, input_image, target_image, input_name, target_name