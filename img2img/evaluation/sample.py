from img2img.utils.diffusion.sampling import sample_image

def get_fake_target(model, cfg, args, inputs, device):
    if cfg['model']['name'] == 'gan':
        fake_targets = model(inputs)
    elif cfg['model']['name'] == 'diffusion':
        fake_targets = sample_image(
            config=cfg,
            model=model,
            input_image=inputs,
            initial_noise=None,
            device=device,
            create_list=False,
            args=args
        )
    return fake_targets