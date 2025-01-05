""" 
Adapted from:
    https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
    https://github.com/mirthAI/Fast-DDPM/blob/main/runners/diffusion.py
"""

import numpy as np

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# Test =====================================================================================
if __name__ == "__main__":
    import torch
    import yaml
    import matplotlib.pyplot as plt
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open("configs/sdo/ddpm.yaml", "r") as f:
        config = yaml.safe_load(f)

    betas = get_beta_schedule(
        beta_schedule=config['params']['diffusion']['beta_schedule'],
        beta_start=config['params']['diffusion']['beta_start'],
        beta_end=config['params']['diffusion']['beta_end'],
        num_diffusion_timesteps=config['params']['diffusion']['num_timesteps']
    )
    betas = torch.from_numpy(betas).float().to(device)

    print("Betas")
    print(betas.shape)
    # plt.plot(betas.cpu())
    # plt.show()

    alphas = (1-betas).cumprod(dim=0)
    print("Alphas")
    print(alphas.shape)
    # plt.plot(alphas.cpu())
    # plt.show()

    t = torch.IntTensor([0]).to(device)
    print(alphas.index_select(dim=0, index=t).view(-1, 1, 1, 1))