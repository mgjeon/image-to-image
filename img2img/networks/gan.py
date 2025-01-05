from .pix2pix.networks_simple import SimpleGenerator, SimpleDiscriminator
from .pix2pix.networks_pix2pix import get_norm_layer, init_weights, ResnetGenerator, UnetGenerator, NLayerDiscriminator, PixelDiscriminator
import img2img.networks.pix2pixhd.networks_default_hd as pix2pixhd
import img2img.networks.pix2pixcc.networks_default_cc as pix2pixcc

# =============================================================================
def define_G(model_cfg):
    name_G = model_cfg['generator']['name']
    args = model_cfg['generator']['args']

    if name_G == 'simple_G':
        net_G = SimpleGenerator(
            in_channels=args['in_channels'],
            out_channels=args['out_channels'],
            features=args['features']
        )
    elif name_G == 'resnet':
        net_G = ResnetGenerator(
            input_nc=args['input_nc'],
            output_nc=args['output_nc'],
            ngf=args['ngf'],
            norm_layer=get_norm_layer(norm_type=args['norm_layer']),
            use_dropout=args['use_dropout'],
            n_blocks=args['n_blocks'],
            padding_type=args['padding_type']
        )
        init_weights(net_G, init_type=args['init_type'], init_gain=args['init_gain'])
    elif name_G == 'unet':
        net_G = UnetGenerator(
            input_nc=args['input_nc'],
            output_nc=args['output_nc'],
            num_downs=args['num_downs'],
            ngf=args['ngf'],
            norm_layer=get_norm_layer(norm_type=args['norm_layer']),
            use_dropout=args['use_dropout']
        )
        init_weights(net_G, init_type=args['init_type'], init_gain=args['init_gain'])
    elif name_G == 'pix2pixhd':
        net_G = pix2pixhd.Generator(
            input_ch=args['input_ch'],
            target_ch=args['target_ch'],
            n_gf=args['n_gf'],
            n_downsample=args['n_downsample'],
            n_residual=args['n_residual'],
            norm_type=args['norm_type'],
            padding_type=args['padding_type'],
        )
        net_G.apply(pix2pixhd.weights_init)
    elif name_G == 'pix2pixcc':
        net_G = pix2pixcc.Generator(
            input_ch=args['input_ch'],
            target_ch=args['target_ch'],
            n_gf=args['n_gf'],
            n_downsample=args['n_downsample'],
            n_residual=args['n_residual'],
            norm_type=args['norm_type'],
            padding_type=args['padding_type'],
            trans_conv=args['trans_conv']
        )
        net_G.apply(pix2pixcc.weights_init)
    else:
        raise NotImplementedError(f"Generator model name '{name_G}' is not implemented")
    
    return net_G


def define_D(model_cfg):
    name_D = model_cfg['discriminator']['name']
    args = model_cfg['discriminator']['args']

    if name_D == 'simple_D':
        net_D = SimpleDiscriminator(
            in_channels=args['in_channels'],
            features=args['features']
        )
    elif name_D == 'patchgan':
        net_D = NLayerDiscriminator(
            input_nc=args['input_nc'],
            ndf=args['ndf'],
            n_layers=args['n_layers'],
            norm_layer=get_norm_layer(norm_type=args['norm_layer'])
        )
        init_weights(net_D, init_type=args['init_type'], init_gain=args['init_gain'])
    elif name_D == 'pixel':
        net_D = PixelDiscriminator(
            input_nc=args['input_nc'],
            ndf=args['ndf'],
            norm_layer=get_norm_layer(norm_type=args['norm_layer'])
        )
        init_weights(net_D, init_type=args['init_type'], init_gain=args['init_gain'])
    elif name_D == 'pix2pixhd':
        net_D = pix2pixhd.Discriminator(
            input_ch=args['input_ch'],
            target_ch=args['target_ch'],
            n_df=args['n_df'],
            n_D=args['n_D'],
        )
        net_D.apply(pix2pixhd.weights_init)
    elif name_D == 'pix2pixcc':
        net_D = pix2pixcc.Discriminator(
            input_ch=args['input_ch'],
            target_ch=args['target_ch'],
            n_df=args['n_df'],
            ch_balance=args['ch_balance'],
            n_D=args['n_D'],
        )
        net_D.apply(pix2pixcc.weights_init)
    else:
        raise NotImplementedError(f"Discriminator model name '{name_D}' is not implemented")
    
    return net_D


