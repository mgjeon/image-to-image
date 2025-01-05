"""
Adapted from:
    https://github.com/JeongHyunJin/Pix2PixHD
"""

import torch
from torch import nn


class Loss:
    def __init__(self, cfg):
        self.cfg = cfg 
        loss_cfg = cfg['params']['loss']

        if loss_cfg['criterion']['name'] == 'MSELoss':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {self.loss_rec_name} not implemented")
        
        if loss_cfg['FMcriterion']['name'] == 'L1Loss':
            self.FMcriterion = nn.L1Loss()
        else:
            raise NotImplementedError(f"Loss {self.loss_rec_name} not implemented")
        
        self.n_D = cfg['model']['discriminator']['args']['n_D']
        self.lambda_FM = loss_cfg['FMcriterion']['args']['lambda_FM']
        
    def __call__(self, G, D, inputs, real_targets):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake_targets = G(inputs)

        # Discriminator loss
        real_features = D(torch.cat((inputs, real_targets), dim=1))
        fake_features = D(torch.cat((inputs, fake_targets.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = torch.ones_like(real_features[i][-1], device=real_features[i][-1].device)
            fake_grid = torch.zeros_like(fake_features[i][-1], device=fake_features[i][-1].device)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5
        
        # Generator loss
        fake_features_pool = D(torch.cat((inputs, fake_targets), dim=1))

        for i in range(self.n_D):
            real_grid = torch.ones_like(fake_features_pool[i][-1], device=fake_features_pool[i][-1].device)
            loss_G += self.criterion(fake_features_pool[i][-1], real_grid)
            
            for j in range(len(fake_features_pool[i])):
                loss_G_FM += self.FMcriterion(fake_features_pool[i][j], real_features[i][j].detach())
                
            loss_G += loss_G_FM * (1.0 / self.n_D) * self.lambda_FM

        return loss_G, loss_D, fake_targets