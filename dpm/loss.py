from torch import nn

class Loss:
    def __init__(self, cfg):
        self.cfg = cfg 

        loss_name = cfg['params']['loss']['name']
        self.loss_args = cfg['params']['loss']['args']
        if loss_name == 'MSELoss':
            self.loss = nn.MSELoss(reduction=self.loss_args['reduction'])
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")
        

    def __call__(self, true_noise, pred_noise):
        # [B, C, H, W]
        n = true_noise.shape[0]  # batch size

        loss = self.loss(true_noise, pred_noise)
        if self.loss_args['average_over_batch'] and self.loss_args['reduction'] == 'sum':
            loss = loss / n

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

    

    lossfn = Loss(config)
    true_noise, pred_noise = true_noise.to(device), pred_noise.to(device)
    loss = lossfn(true_noise, pred_noise)
    print("Loss")
    print(loss)
    print("average_over_batch")
    print(lossfn.loss_args['average_over_batch'])