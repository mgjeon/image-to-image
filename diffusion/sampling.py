import torch
from noise_schedule import get_beta_schedule
from tqdm import tqdm

def compute_alpha(betas, t):
    # [0, betas]
    betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    # [1, alphas_cumprod]
    a = (1 - betas).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def sample_image(*, config, model, input_image, initial_noise=None, device=None, create_list=False):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = device

    num_timesteps = config['params']['diffusion']['num_timesteps']
    timesteps = config['params']['sampling'].get('timesteps', num_timesteps)
    betas = get_beta_schedule(
        beta_schedule=config['params']['diffusion']['beta_schedule'],
        beta_start=config['params']['diffusion']['beta_start'],
        beta_end=config['params']['diffusion']['beta_end'],
        num_diffusion_timesteps=num_timesteps
    )
    betas = torch.from_numpy(betas).float().to(device)

    skip = num_timesteps // timesteps
    seq = range(0, num_timesteps, skip)
    # print(f"Skip Step: {skip}")
    # print(f"Total Steps: {len(seq)}")

    # [B, C, H, W]
    x_input = input_image.to(device)
    n = x_input.shape[0]

    if initial_noise is None:
        xt = torch.randn(
            n,
            config['model']['output_nc'],
            config['data']['image_size'],
            config['data']['image_size'],
            device=device
        )
        if config['params']['diffusion']['clip_noise']:
            xt = xt.clamp(-1, 1)
    else:
        xt = initial_noise.clone().to(device)

    seq_next = [-1] + list(seq[:-1])
    eta = config['params']['sampling'].get('eta', 0.0)
    pred_type = config['params']['pred']
    # print(f"eta: {eta}")

    if create_list:
        xs = [xt]
        x0_preds = []
    
    model.eval()
    with torch.no_grad():
        for i, j in zip(tqdm(reversed(seq)), reversed(seq_next)):
            t = (torch.ones(n) * i).to(device)
            t_next = (torch.ones(n) * j).to(device)
            at = compute_alpha(betas, t.long())
            at_next = compute_alpha(betas, t_next.long())

            if create_list:
                xt = xs[-1].to(device)

            if pred_type == 'noise':
                et = model(torch.cat([x_input, xt], dim=1), t)
                x0_t = (xt - et*(1.0 - at).sqrt()) / at.sqrt()

            elif pred_type == 'x0':
                x0_t = model(torch.cat([x_input, xt], dim=1), t)
                et = (xt - x0_t*at.sqrt()) / (1.0 - at).sqrt()

            if create_list:
                x0_preds.append(x0_t.to('cpu')) 

            # direction pointing to x_t
            # eta = 0 : DDIM (c1=0)
            # eta = 1 : DDPM (c1!=0)
            c1 = (
                    eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et

            if create_list:
                xs.append(xt_next.to('cpu'))
            else:
                xt = xt_next

    if create_list:
        return xs, x0_preds
    else:
        return xt
    
# Test ========================================================================
if __name__ == "__main__":
    import torch
    import yaml
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open("configs/sdo/ddpm.yaml", "r") as f:
        config = yaml.safe_load(f)

    from networks import define_model
    model = define_model(config).to(device)

    batch_size = 1
    in_channels = 2
    out_channels = 1
    height = 256
    width = 256

    input_image  = torch.randn((batch_size,  in_channels, height, width))   # Input
    initial_noise = torch.randn((batch_size, out_channels, height, width))  # Target
    input_image = input_image.clamp(-1, 1)
    initial_noise = initial_noise.clamp(-1, 1)
    print("Input")
    print(input_image.shape)
    print("Initial Noise (Target)")
    print(initial_noise.shape)

    output_image = sample_image(
        config=config,
        model=model,
        input_image=input_image,
        initial_noise=initial_noise,
        device=device
    )
    print("Output")
    print(output_image.shape)

    xs, x0_preds = sample_image(
        config=config,
        model=model,
        input_image=input_image,
        initial_noise=initial_noise,
        device=device,
        create_list=True
    )
    print("Output List")
    print(len(xs))
    print("x0_preds List")
    print(len(x0_preds))
    print("Output")
    print(xs[-1].shape)

    print(torch.allclose(output_image.cpu(), xs[-1].cpu()))