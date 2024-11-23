from networks_default import Generator, Discriminator, weights_init


# =============================================================================
def define_G(model_cfg):
    name_G = model_cfg['generator']['name']
    args = model_cfg['generator']['args']

    if name_G == 'default_generator':
        net_G = Generator(
            input_ch=args['input_ch'],
            target_ch=args['target_ch'],
            n_gf=args['n_gf'],
            n_downsample=args['n_downsample'],
            n_residual=args['n_residual'],
            norm_type=args['norm_type'],
            padding_type=args['padding_type'],
        )
        net_G.apply(weights_init)
    else:
        raise NotImplementedError(f"Generator model name '{name_G}' is not implemented")
    
    return net_G


def define_D(model_cfg):
    name_D = model_cfg['discriminator']['name']
    args = model_cfg['discriminator']['args']

    if name_D == 'default_discriminator':
        net_D = Discriminator(
            input_ch=args['input_ch'],
            target_ch=args['target_ch'],
            n_df=args['n_df'],
            n_D=args['n_D'],
        )
        net_D.apply(weights_init)
    else:
        raise NotImplementedError(f"Discriminator model name '{name_D}' is not implemented")
    
    return net_D


