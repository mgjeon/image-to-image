"""
Adapted from:
    https://github.com/JeongHyunJin/Pix2PixCC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# Helper functions =============================================================
# Set the initial conditions of weights
def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)


# Set the Normalization method for the input layer
def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d,affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer


# Set the Padding method for the input layer
def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer


# Generator ====================================================================
class Generator(nn.Module):
    def __init__(
        self,
        input_ch=1,                 # The number of channels of input data
        target_ch=1,                # The number of channels of target data
        n_gf=64,                    # The number of channels in the first convolutional layer of the Generator
        n_downsample=4,             # how many times you want to downsample input data in Generator
        n_residual=9,               # the number of residual blocks in Generator
        norm_type='InstanceNorm2d', # normalization type ['BatchNorm2d', 'InstanceNorm2d']
        padding_type='reflection',  # padding type ['reflection', 'replication', 'zero']
        trans_conv=True,            # using transposed convolutions in Generator
    ):
        super(Generator, self).__init__()
        
        #----------------------------------------------------------------------        
        input_ch = input_ch
        output_ch = target_ch
        n_gf = n_gf
        norm = get_norm_layer(norm_type)
        act = Mish()
        pad = get_pad_layer(padding_type)
        trans_conv = trans_conv

        #----------------------------------------------------------------------
        model = []
        
        model += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0), norm(n_gf), act]

        for _ in range(n_downsample):
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=5, padding=2, stride=2), norm(2 * n_gf), act]
            n_gf *= 2

        for _ in range(n_residual):
            model += [ResidualBlock(n_gf, pad, norm, act)]

        for n_up in range(n_downsample):
            #------------------------------------------------------------------
            if trans_conv == True:
                model += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                          norm(n_gf//2), act]
            else:
                model += [nn.UpsamplingBilinear2d(scale_factor=2)]
                model += [pad(1), nn.Conv2d(n_gf, n_gf//2, kernel_size=3, padding=0, stride=1), norm(n_gf//2), act]
            #------------------------------------------------------------------                        
            n_gf //= 2
        
        
        model += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)
        #----------------------------------------------------------------------
        
        # print(self)
        # print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def forward(self, x):
        return self.model(x)

#-----------------------------------------------------------------------------

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
#------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


# Discriminator =================================================================
class PatchDiscriminator(nn.Module):
    def __init__(self, input_ch, target_ch, n_df, ch_balance):
        super(PatchDiscriminator, self).__init__()

        #----------------------------------------------------------------------
        if ch_balance > 0:
            ch_ratio = float(input_ch)/float(target_ch)
            ch_ratio *= ch_balance
            if ch_ratio > 1:
                input_channel = input_ch + target_ch*int(ch_ratio)                            
            elif ch_ratio < 1:
                input_channel = input_ch*int(1/ch_ratio) + target_ch
            else:
                input_channel = input_ch + target_ch
        else:
            input_channel = input_ch + target_ch
        
        #----------------------------------------------------------------------
        act = nn.LeakyReLU(0.2, inplace=True)
        n_df = n_df
        norm = nn.InstanceNorm2d
        
        #----------------------------------------------------------------------
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))
            
        #----------------------------------------------------------------------

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


class Discriminator(nn.Module):
    def __init__(
        self,
        input_ch=1,      # number of input channels
        target_ch=1,     # number of target channels
        n_df=64,         # the number of channels in the first layer of D
        ch_balance=1,    # Set channel balance of input and target data
        n_D=2,           # how many discriminators in differet scales you want to use
    ):
        super(Discriminator, self).__init__()

        #----------------------------------------------------------------------
        for i in range(n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(
                input_ch=input_ch,
                target_ch=target_ch,
                n_df=n_df,
                ch_balance=ch_balance,
            ))
        self.n_D = n_D

        #----------------------------------------------------------------------
        # print(self)
        # print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
                
        return result



# Test =========================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 1
    in_channels = 2
    out_channels = 1
    height = 256
    width = 256
    inputs = torch.randn((batch_size, in_channels, height, width)).to(device)  # Input
    real_targets = torch.randn((batch_size, out_channels, height, width)).to(device)  # Target
    print("Input")
    print(inputs.shape)
    print("Target")
    print(real_targets.shape)

    net_G = Generator(
        input_ch=in_channels,
        target_ch=out_channels,
    ).apply(weights_init).to(device)
    n_D=2
    net_D = Discriminator(
        input_ch=in_channels,
        target_ch=out_channels,
        n_D=n_D,
    ).apply(weights_init).to(device)

    fake_targets = net_G(inputs)
    print("Generator")
    print(fake_targets.shape)

    ch_balance = 1
    if ch_balance > 0:
        real_pair = torch.cat([inputs, real_targets], dim=1)
        fake_pair = torch.cat([inputs, fake_targets.detach()], dim=1)

        ch_plus = 0
        ch_ratio = float(in_channels)/float(out_channels)
        ch_ratio *= ch_balance
        if ch_ratio > 1:
            for dr in range(int(ch_ratio)-1):
                real_pair = torch.cat((real_pair, real_targets), dim=1)
                fake_pair = torch.cat((fake_pair, fake_targets.detach()), dim=1)
                ch_plus += out_channels
        
        elif ch_ratio < 1:
            for dr in range(int(1/ch_ratio)-1):
                real_pair = torch.cat((inputs, real_pair), dim=1)
                fake_pair = torch.cat((inputs, fake_pair), dim=1)
                ch_plus += in_channels

        else:
            pass

        print("Real Pair")
        print(real_pair.shape)
        print("Fake Pair")
        print(fake_pair.shape)

        real_features = net_D(real_pair)
        fake_features = net_D(fake_pair)
    else:
        real_features = net_D(torch.cat((inputs, fake_targets), dim=1))
        fake_features = net_D(torch.cat((inputs, fake_targets.detach()), dim=1))

    print("Discriminator")
    print(len(real_features))
    for i in range(n_D):
        print(i, len(real_features[i]), real_features[i][-1].shape)
        for j in range(len(real_features[i])):
            print(j, real_features[i][j].shape)