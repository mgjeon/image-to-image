import torch
from torch import nn


class Loss:
    def __init__(self, cfg):
        self.cfg = cfg 

        self.loss_rec_name = cfg['params']['loss']['reconstruction']['name']
        self.loss_rec_args = cfg['params']['loss']['reconstruction']['args']
        if self.loss_rec_name == 'L1Loss':
            self.loss_rec = nn.L1Loss()
        else:
            raise NotImplementedError(f"Loss {self.loss_rec_name} not implemented")
        
        loss_adv_name = cfg['params']['loss']['adversarial']['name']
        if loss_adv_name == 'BCEWithLogitsLoss':
            self.loss_adv = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"Loss {loss_adv_name} not implemented")
        
    def __call__(self, G, D, inputs, real_targets):

        fake_targets = G(inputs)

        # Generator loss
        pred_fake = D(torch.cat((inputs, fake_targets), dim=1))
        loss_G_adv = self.loss_adv(pred_fake, torch.ones_like(pred_fake))
        if self.loss_rec_name == 'L1Loss':
            lambda_L1 = self.loss_rec_args['lambdaL1']
            loss_G_rec = lambda_L1 * self.loss_rec(real_targets, fake_targets)
        else:
            raise NotImplementedError(f"Loss {self.loss_rec_name} not implemented")
        loss_G = loss_G_adv + loss_G_rec

        # Discriminator loss
        pred_real = D(torch.cat((inputs, real_targets), dim=1))
        loss_D_adv_real = self.loss_adv(pred_real, torch.ones_like(pred_real))
        pred_fake = D(torch.cat((inputs, fake_targets.detach()), dim=1))
        loss_D_adv_fake = self.loss_adv(pred_fake, torch.zeros_like(pred_fake))
        loss_D = 0.5 * (loss_D_adv_real + loss_D_adv_fake)

        return loss_G, loss_D, fake_targets