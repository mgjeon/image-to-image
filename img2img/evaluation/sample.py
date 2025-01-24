from img2img.utils.diffusion.sampling import sample_image

def get_fake_target(model, cfg, args, inputs, device):
    if cfg['model']['name'] == 'gan':
        fake_targets = model(inputs)
    elif cfg['model']['name'] == 'diffusion':
        output = sample_image(
            config=cfg,
            model=model,
            input_image=inputs,
            initial_noise=None,
            device=device,
            create_list=False,
            args=args,
            return_seq=args.return_seq
        )
        if args.return_seq:
            fake_targets, seq, timesteps = output
        else:
            fake_targets = output

    res = {}
    res['fake_targets'] = fake_targets
    try:
        if args.return_seq:
            res['seq'] = seq
            res['timesteps'] = timesteps
    except:
        pass
    return res