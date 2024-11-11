from networks_ddim import Model

# =============================================================================
def define_model(cfg):
    name = cfg['model']['name']
    args = cfg['model']['args']
    image_size = cfg['data']['image_size']
    # num_timesteps = cfg['params']['diffusion']['num_timesteps']

    if name == 'ddim_unet':
        model = Model(
            input_nc=args['input_nc'],
            output_nc=args['output_nc'],
            ch=args['ch'],
            ch_mult=args['ch_mult'],
            num_res_blocks=args['num_res_blocks'],
            attn_resolutions=args['attn_resolutions'],
            dropout=args['dropout'],
            resamp_with_conv=args['resamp_with_conv'],
            resolution=image_size,
            # num_timesteps=num_timesteps
        )
    else:
        raise NotImplementedError(f"Model name '{name}' is not implemented")
    
    return model


# Test ========================================================================
if __name__ == "__main__":
    import torch
    import yaml
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open("configs/sdo/ddpm.yaml", "r") as f:
        config = yaml.safe_load(f)

    batch_size = 1
    in_channels = 2
    out_channels = 1
    height = 256
    width = 256

    real_input  = torch.randn((batch_size,  in_channels, height, width))  # Input
    real_target = torch.randn((batch_size, out_channels, height, width))  # Target
    print("Input")
    print(real_input.shape)
    print("Target")
    print(real_target.shape)

    model = define_model(config).to(device)
    t = torch.IntTensor([100]).reshape((1,)).to(device)
    real_input, real_target = real_input.to(device), real_target.to(device)
    fake_target = model(torch.cat([real_input, real_target], dim=1), t.float())
    print("Generator")
    print(fake_target.shape)
    assert fake_target.shape == real_target.shape