"""
Adapted from:
    https://github.com/JeongHyunJin/Pix2PixCC
"""

import torch
from torch import nn


class Loss:
    def __init__(self, cfg):
        self.cfg = cfg 
        loss_cfg = cfg['params']['loss']

        if loss_cfg['LSGANcriterion']['name'] == 'MSELoss':
            self.LSGANcriterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss {self.loss_rec_name} not implemented")
        
        if loss_cfg['FMcriterion']['name'] == 'L1Loss':
            self.FMcriterion = nn.L1Loss()
        else:
            raise NotImplementedError(f"Loss {self.loss_rec_name} not implemented")
        
        self.input_ch = cfg['model']['discriminator']['args']['input_ch']
        self.target_ch = cfg['model']['discriminator']['args']['target_ch']
        self.ch_balance = cfg['model']['discriminator']['args']['ch_balance']
        self.n_D = cfg['model']['discriminator']['args']['n_D']

        self.lambda_LSGAN = loss_cfg['LSGANcriterion']['args']['lambda_LSGAN']
        self.lambda_FM = loss_cfg['FMcriterion']['args']['lambda_FM']
        self.n_CC = loss_cfg['CCcriterion']['args']['n_CC']
        self.lambda_CC = loss_cfg['CCcriterion']['args']['lambda_CC']
        
    def __call__(self, G, D, inputs, real_targets):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake_targets = G(inputs)

        #----------------------------------------------------------------------
        # [1] Get Real and Fake (Generated) pairs and features 

        if self.ch_balance > 0:
            
            real_pair = torch.cat((inputs, real_targets), dim=1)
            fake_pair = torch.cat((inputs, fake_targets.detach()), dim=1)

            ch_plus = 0
            ch_ratio = float(self.input_ch)/float(self.target_ch)
            ch_ratio *= self.ch_balance
            if ch_ratio > 1:
                for _ in range(int(ch_ratio)-1):
                    real_pair = torch.cat((real_pair, real_targets), dim=1)
                    fake_pair = torch.cat((fake_pair, fake_targets.detach()), dim=1)
                    ch_plus += self.target_ch
            
            elif ch_ratio < 1:
                for _ in range(int(1/ch_ratio)-1):
                    real_pair = torch.cat((inputs, real_pair), dim=1)
                    fake_pair = torch.cat((inputs, fake_pair), dim=1)
                    ch_plus += self.input_ch
            
            else:
                pass
                
            real_features = D(real_pair)
            fake_features = D(fake_pair)
        else:
            real_features = D(torch.cat((inputs, real_targets), dim=1))
            fake_features = D(torch.cat((inputs, fake_targets.detach()), dim=1))

        #----------------------------------------------------------------------
        # [2] Compute LSGAN loss for the discriminator

        for i in range(self.n_D):
            real_grid = torch.ones_like(real_features[i][-1], device=real_features[i][-1].device)
            fake_grid = torch.zeros_like(fake_features[i][-1], device=fake_features[i][-1].device)

            loss_D += (self.LSGANcriterion(real_features[i][-1], real_grid) +
                       self.LSGANcriterion(fake_features[i][-1], fake_grid)) * 0.5
        
        #----------------------------------------------------------------------
        # [3] Compute LSGAN loss and Feature Matching loss for the generator

        if self.ch_balance > 0:
            fake_pair_pool = torch.cat((inputs, fake_targets), dim=1)

            if ch_ratio > 1:
                for _ in range(int(ch_ratio)-1):
                    fake_pair_pool = torch.cat((fake_pair_pool, fake_targets), dim=1)
            elif ch_ratio < 1:
                for _ in range(int(1/ch_ratio)-1):
                    fake_pair_pool = torch.cat((inputs, fake_pair_pool), dim=1)
            else:
                pass
        
            fake_features_pool = D(fake_pair_pool)
        else:
            fake_features_pool = D(torch.cat((inputs, fake_targets), dim=1))

        for i in range(self.n_D):
            real_grid = torch.ones_like(fake_features_pool[i][-1], device=fake_features_pool[i][-1].device)
            loss_G += self.LSGANcriterion(fake_features_pool[i][-1], real_grid) * 0.5 * self.lambda_LSGAN
            
            for j in range(len(fake_features_pool[i])):
                loss_G_FM += self.FMcriterion(fake_features_pool[i][j], real_features[i][j].detach())
            
            loss_G += loss_G_FM * (1.0 / self.n_D) * self.lambda_FM
        
        #----------------------------------------------------------------------
        # [4] Compute Correlation Coefficient loss for the generator

        for i in range(self.n_CC):
            real_down = real_targets
            fake_down = fake_targets
            for _ in range(i):
                real_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(real_down)
                fake_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(fake_down)
            
            loss_CC = self.Inspector(real_down, fake_down)
            loss_G += loss_CC * (1.0 / self.n_CC) * self.lambda_CC

        #----------------------------------------------------------------------
        return loss_G, loss_D
    
# Inspector ====================================================================
    def Inspector(self, real, fake):

        rd = real - torch.mean(real)
        fd = fake - torch.mean(fake)

        r_num = torch.sum(rd * fd)
        r_den = torch.sqrt(torch.sum(rd ** 2)) * torch.sqrt(torch.sum(fd ** 2))
        if r_den.dtype == torch.float16:
            eps = 1e-4
        elif r_den.dtype == torch.float32:
            eps = 1e-8
        else:
            eps = 1e-8
        PCC_val = r_num / (r_den + eps)

        #----------------------------------------------------------------------
        cc_name = self.cfg['params']['loss']['CCcriterion']['name']
        if cc_name == 'Concordance':
            numerator = 2*PCC_val*torch.std(real)*torch.std(fake)
            denominator = (torch.var(real) + torch.var(fake)
                           + (torch.mean(real) - torch.mean(fake))**2)
            
            CCC_val = numerator/(denominator + eps)
            loss_CC = (1.0 - CCC_val)
            
        elif cc_name == 'Pearson':
            loss_CC = (1.0 - PCC_val)
        else:
            raise NotImplementedError(f"Loss {cc_name} not implemented")
        
        #----------------------------------------------------------------------
        return loss_CC