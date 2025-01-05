from torch import nn

def define_loss(cfg):
    pred_type = cfg['params']['pred']

    if pred_type == 'noise':
        return NoiseLoss(cfg)
    elif pred_type == 'x0':
        return X0Loss(cfg)
    elif pred_type == 'both':
        return BothLoss(cfg)
    else:
        raise NotImplementedError(f"Prediction type {pred_type} not implemented")


class NoiseLoss:
    def __init__(self, cfg):
        self.cfg = cfg 

        loss_name = cfg['params']['loss']['name']
        self.loss_args = cfg['params']['loss']['args']
        if loss_name == 'MSELoss':
            self.loss = nn.MSELoss(reduction=self.loss_args['reduction'])
        elif loss_name == 'L1Loss':
            self.loss = nn.L1Loss(reduction=self.loss_args['reduction'])
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")
        

    def __call__(self, true_noise, pred_noise):
        loss = self.loss(true_noise, pred_noise)

        if self.loss_args.get('average_over_batch', False) and self.loss_args['reduction'] == 'sum':
            # [B, C, H, W]
            n = true_noise.shape[0]  # batch size
            loss = loss / n

        return loss
    


class X0Loss:
    def __init__(self, cfg):
        self.cfg = cfg 

        loss_name = cfg['params']['loss']['name']
        self.loss_args = cfg['params']['loss']['args']
        if loss_name == 'MSELoss':
            self.loss = nn.MSELoss(reduction=self.loss_args['reduction'])
        elif loss_name == 'L1Loss':
            self.loss = nn.L1Loss(reduction=self.loss_args['reduction'])
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")
        

    def __call__(self, true_x0, pred_x0):
        loss = self.loss(true_x0, pred_x0)

        if self.loss_args.get('average_over_batch', False) and self.loss_args['reduction'] == 'sum':
            # [B, C, H, W]
            n = true_x0.shape[0]  # batch size
            loss = loss / n

        return loss
    

class BothLoss:
    def __init__(self, cfg):
        self.cfg = cfg

        noise_loss_name = cfg['params']['loss']['noise']['name']
        self.noise_loss_args = cfg['params']['loss']['noise']['args']
        if noise_loss_name == 'MSELoss':
            self.noise_loss = nn.MSELoss(reduction=self.noise_loss_args['reduction'])
        elif noise_loss_name == 'L1Loss':
            self.noise_loss = nn.L1Loss(reduction=self.noise_loss_args['reduction'])
        else:
            raise NotImplementedError(f"Loss {noise_loss_name} not implemented")

        x0_loss_name = cfg['params']['loss']['x0']['name']
        self.x0_loss_args = cfg['params']['loss']['x0']['args']
        if x0_loss_name == 'MSELoss':
            self.x0_loss = nn.MSELoss(reduction=self.x0_loss_args['reduction'])
        elif x0_loss_name == 'L1Loss':
            self.x0_loss = nn.L1Loss(reduction=self.x0_loss_args['reduction'])
        else:
            raise NotImplementedError(f"Loss {x0_loss_name} not implemented")

    def __call__(self, true_noise, pred_noise, true_x0, pred_x0):
        
        noise_loss = self.noise_loss(true_noise, pred_noise)
        x0_loss = self.x0_loss(true_x0, pred_x0)

        if self.noise_loss_args.get('average_over_batch', False) and self.noise_loss_args['reduction'] == 'sum':
            # [B, C, H, W]
            n = true_noise.shape[0]
            noise_loss = noise_loss / n
        
        if self.x0_loss_args.get('average_over_batch', False) and self.x0_loss_args['reduction'] == 'sum':
            # [B, C, H, W]
            n = true_x0.shape[0]
            x0_loss = x0_loss / n

        loss = noise_loss + x0_loss
        
        return loss
    
# Test ========================================================================
if __name__ == "__main__":
    import torch
    import yaml
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open("configs/sdo/ddpm.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    batch_size = 2
    out_channels = 1
    height = 256
    width = 256

    true_noise  = torch.randn((batch_size, out_channels, height, width))  # Input
    pred_noise = torch.randn((batch_size, out_channels, height, width))  # Target
    print("True Noise")
    print(true_noise.shape)
    print("Pred Noise")
    print(pred_noise.shape)

    

    lossfn = NoiseLoss(config)
    true_noise, pred_noise = true_noise.to(device), pred_noise.to(device)
    loss = lossfn(true_noise, pred_noise)
    print("Loss")
    print(loss)
    print("average_over_batch")
    print(lossfn.loss_args['average_over_batch'])