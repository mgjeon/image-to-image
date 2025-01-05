import lightning as L
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from img2img.data.dataset import AlignedDataset
from img2img.networks.gan import define_G, define_D
from ema_pytorch import EMA


# LightningModule =================================================================
class GAN(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Multi-optimizer training

# Create output directory =========================================================
        output_dir = Path(cfg['params']['output_dir'])
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

# Model ===========================================================================
        self.cfg = cfg
        self.net_G = define_G(cfg['model'])
        self.net_D = define_D(cfg['model'])

# Loss ============================================================================
        if cfg['params']['loss']['name'] == 'pix2pix':
            from img2img.loss.pix2pix import Loss
            print("Using pix2pix loss")
        elif cfg['params']['loss']['name'] == 'pix2pixhd':
            from img2img.loss.pix2pixhd import Loss
            print("Using pix2pixHD loss")
        elif cfg['params']['loss']['name'] == 'pix2pixcc':
            from img2img.loss.pix2pixcc import Loss
            print("Using pix2pixCC loss")
        self.criterion = Loss(cfg)

# EMA ============================================================================
        if cfg['params']['ema']['active']:
            self.ema = EMA(self.net_G, beta=cfg['params']['ema']['rate'])
        else:
            self.ema = None

# Metrics =========================================================================
        self.train_mae = MeanAbsoluteError()
        self.train_pcc = PearsonCorrCoef()
        self.train_psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

        self.val_mae = MeanAbsoluteError()
        self.val_pcc = PearsonCorrCoef()
        self.val_psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

        self.test_mae = MeanAbsoluteError()
        self.test_pcc = PearsonCorrCoef()
        self.test_psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

        # if self.ema is not None:
        #     self.train_mae_ema = MeanAbsoluteError()
        #     self.train_pcc_ema = PearsonCorrCoef()
        #     self.val_mae_ema = MeanAbsoluteError()
        #     self.val_pcc_ema = PearsonCorrCoef()
            
# Dataset =========================================================================
    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = AlignedDataset(
                input_dir=self.cfg['data']['train']['input_dir'],
                target_dir=self.cfg['data']['train']['target_dir'],
                image_size=self.cfg['data']['image_size'],
                ext=self.cfg['data']['ext'],
            )
            self.val_dataset = AlignedDataset(
                input_dir=self.cfg['data']['val']['input_dir'],
                target_dir=self.cfg['data']['val']['target_dir'],
                image_size=self.cfg['data']['image_size'],
                ext=self.cfg['data']['ext'],
            )
        
        if stage == 'test':
            self.test_dataset = AlignedDataset(
                input_dir=self.cfg['data']['test']['input_dir'],
                target_dir=self.cfg['data']['test']['target_dir'],
                image_size=self.cfg['data']['image_size'],
                ext=self.cfg['data']['ext'],
            )
        

# DataLoader ======================================================================
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg['data']['train']['batch_size'],
            shuffle=self.cfg['data']['train']['shuffle'],
            num_workers=self.cfg['data']['train']['num_workers'],
            pin_memory=self.cfg['data']['train']['pin_memory'],
            drop_last=self.cfg['data']['train']['drop_last'],
            persistent_workers=self.cfg['data']['train']['persistent_workers'],
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg['data']['val']['batch_size'],
            shuffle=self.cfg['data']['val']['shuffle'],
            num_workers=self.cfg['data']['val']['num_workers'],
            pin_memory=self.cfg['data']['val']['pin_memory'],
            drop_last=self.cfg['data']['val']['drop_last'],
            persistent_workers=self.cfg['data']['val']['persistent_workers'],
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg['data']['test']['batch_size'],
            shuffle=self.cfg['data']['test']['shuffle'],
            num_workers=self.cfg['data']['test']['num_workers'],
            pin_memory=self.cfg['data']['test']['pin_memory'],
            drop_last=self.cfg['data']['test']['drop_last'],
            persistent_workers=self.cfg['data']['test']['persistent_workers'],
        )

# Optimizer =======================================================================
    def configure_optimizers(self):
        if self.cfg['params']['optimizer']['name'] == 'Adam':
            args = self.cfg['params']['optimizer']['args']
            optim_G = optim.Adam(self.net_G.parameters(), lr=args['lr'], betas=args['betas'])
            optim_D = optim.Adam(self.net_D.parameters(), lr=args['lr'], betas=args['betas'])
        else:
            raise NotImplementedError(f"Optimizer {self.cfg['params']['optimizer']['name']} not implemented")

        return optim_G, optim_D
    
# Number of parameters ============================================================
    def on_train_start(self):
        num_params_G = sum(p.numel() for p in self.net_G.parameters())
        num_params_D = sum(p.numel() for p in self.net_D.parameters())
        with open(Path(self.loggers[0].log_dir) / 'log.log', 'w') as file:
            file.write(f"Number of parameters G: {num_params_G}\n")
            file.write(f"Number of parameters D: {num_params_D}")

# Training Step ===================================================================
    def training_step(self, batch, batch_idx):
        inputs, real_targets, _, _ = batch

        optim_G, optim_D = self.optimizers()

        loss_G, loss_D, fake_targets = self.criterion(self.net_G, self.net_D, inputs, real_targets)

# Train Generator =================================================================
        optim_G.zero_grad()
        self.manual_backward(loss_G)
        if self.cfg['params']['grad_clip']['clip']:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.net_G.parameters(),
                max_norm=self.cfg['params']['grad_clip']['max_norm']
            )
        optim_G.step()

# Train Discriminator =============================================================
        optim_D.zero_grad()
        self.manual_backward(loss_D)
        if self.cfg['params']['grad_clip']['clip']:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.net_D.parameters(),
                max_norm=self.cfg['params']['grad_clip']['max_norm']
            )
        optim_D.step()

# Log Loss ========================================================================
        self.log_dict({'G_loss_step': loss_G, 'D_loss_step': loss_D}, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log_dict({'G_loss_epoch': loss_G, 'D_loss_epoch': loss_D}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))

# Log Metrics ====================================================================
        fake_targets = fake_targets.detach()
        real_targets = real_targets.detach()
        fake_targets = torch.clamp(fake_targets, min=-1.0, max=1.0)
        real_targets = torch.clamp(real_targets, min=-1.0, max=1.0)
        self.train_mae.update(fake_targets, real_targets)
        self.train_pcc.update(fake_targets.double().flatten(), real_targets.double().flatten())
        self.train_psnr.update(fake_targets, real_targets)
        self.train_ssim.update(fake_targets, real_targets)

        # if self.ema is not None:
        #     fake_targets = self.ema(inputs)
        #     self.train_mae_ema.update(fake_targets, real_targets)
        #     self.train_pcc_ema.update(fake_targets.double().flatten(), real_targets.double().flatten())

#==================================================================================
    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update()

#==================================================================================
    def on_train_epoch_end(self):
        global_step = self.global_step // 2  # self.global_step is the number of optimizer steps taken, in this case 2 steps per iteration because of the multi-optimizer training
        train_metrics = {
            'mae/train': self.train_mae.compute(),
            'pcc/train': self.train_pcc.compute(),
            'psnr/train': self.train_psnr.compute(),
            'ssim/train': self.train_ssim.compute(),
        }
        self.log_dict(train_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.train_mae.reset()
        self.train_pcc.reset()
        self.train_psnr.reset()
        self.train_ssim.reset()
        # if self.ema is not None:
        #     self.log_dict({'mae/train_ema': self.train_mae_ema.compute(), 'pcc/train_ema': self.train_pcc_ema.compute()}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        #     self.train_mae_ema.reset()
        #     self.train_pcc_ema.reset()

# Save Image ======================================================================
        if self.current_epoch % self.cfg['params']['save_img_per_epoch'] == 0:
            output_image_train_dir = Path(self.loggers[0].log_dir)/"images"/"train"
            output_image_val_dir = Path(self.loggers[0].log_dir)/"images"/"val"
            if not output_image_train_dir.exists(): output_image_train_dir.mkdir(parents=True, exist_ok=True)
            if not output_image_val_dir.exists(): output_image_val_dir.mkdir(parents=True, exist_ok=True)
            self.net_G.eval()
            with torch.no_grad():
                for inputs, real_target, _, _ in self.train_dataloader():
                    inputs = inputs.to(self.device)
                    fake_target = self.net_G(inputs)
                    self.train_dataset.save_image(fake_target[0], output_image_train_dir/f"{self.current_epoch}_{global_step-1}_fake.png")
                    self.train_dataset.save_image(real_target[0], output_image_train_dir/f"{self.current_epoch}_{global_step-1}_real.png")
                    train_fig = self.train_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                    self.logger.experiment.add_figure('fig/train', train_fig, global_step-1)

                    # fake_target = self.ema(inputs)
                    # self.train_dataset.save_image(fake_target[0], output_image_train_dir/f"{self.current_epoch}_{global_step-1}_fake_ema.png")
                    # train_fig = self.train_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                    # self.logger.experiment.add_figure('fig/train_ema', train_fig, global_step-1)
                    break
                for inputs, real_target, _, _ in self.val_dataloader():
                    inputs = inputs.to(self.device)
                    fake_target = self.net_G(inputs)
                    self.val_dataset.save_image(fake_target[0], output_image_val_dir/f"{self.current_epoch}_{global_step-1}_fake.png")
                    self.val_dataset.save_image(real_target[0], output_image_val_dir/f"{self.current_epoch}_{global_step-1}_real.png")
                    val_fig = self.val_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                    self.logger.experiment.add_figure('fig/val', val_fig, global_step-1)

                    # fake_target = self.ema(inputs)
                    # self.val_dataset.save_image(fake_target[0], output_image_val_dir/f"{self.current_epoch}_{global_step-1}_fake_ema.png")
                    # val_fig = self.val_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                    # self.logger.experiment.add_figure('fig/val_ema', val_fig, global_step-1)
                    break
            self.net_G.train()

# Save Model =======================================================================
        # if self.current_epoch % self.cfg['params']['save_state_per_epoch'] == 0 or self.current_epoch == self.cfg['params']['num_epochs']-1:
        #     ckpt_path = Path(self.loggers[0].log_dir)/"checkpoints"
        #     if not ckpt_path.exists(): ckpt_path.mkdir(parents=True, exist_ok=True)
        #     torch.save(self.net_G.state_dict(), ckpt_path/f"{self.current_epoch}_{global_step-1}_G.pth")
        #     torch.save(self.ema.ema_model.state_dict(), ckpt_path/f"{self.current_epoch}_{global_step-1}_G_ema.pth")

# Validation Step ==================================================================
    def validation_step(self, batch, batch_idx):
        inputs, real_targets, _, _ = batch

        loss_G, loss_D, fake_targets = self.criterion(self.net_G, self.net_D, inputs, real_targets)

        self.log_dict({'G_loss_val': loss_G, 'D_loss_val': loss_D}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))

        fake_targets = fake_targets.detach()
        real_targets = real_targets.detach()
        fake_targets = torch.clamp(fake_targets, min=-1.0, max=1.0)
        real_targets = torch.clamp(real_targets, min=-1.0, max=1.0)
        self.val_mae.update(fake_targets, real_targets)
        self.val_pcc.update(fake_targets.double().flatten(), real_targets.double().flatten())
        self.val_psnr.update(fake_targets, real_targets)
        self.val_ssim.update(fake_targets, real_targets)
        # if self.ema is not None:
        #     fake_targets = self.ema(inputs)
        #     self.val_mae_ema.update(fake_targets, real_targets)
        #     self.val_pcc_ema.update(fake_targets.double().flatten(), real_targets.flatten())
        
    def on_validation_epoch_end(self):
        val_metrics = {
            'mae/val': self.val_mae.compute(),
            'pcc/val': self.val_pcc.compute(),
            'psnr/val': self.val_psnr.compute(),
            'ssim/val': self.val_ssim.compute(),
        }
        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_mae.reset()
        self.val_pcc.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()
        # if self.ema is not None:
        #     self.log_dict({'mae/val_ema': self.val_mae_ema.compute(), 'pcc/val_ema': self.val_pcc_ema.compute()}, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #     self.val_mae_ema.reset()
        #     self.val_pcc_ema.reset()

# Test Step =======================================================================
    def test_step(self, batch, batch_idx):
        inputs, real_targets, _, _ = batch
        fake_targets = self.net_G(inputs)

        fake_targets = fake_targets.detach()
        real_targets = real_targets.detach()
        fake_targets = torch.clamp(fake_targets, min=-1.0, max=1.0)
        real_targets = torch.clamp(real_targets, min=-1.0, max=1.0)
        self.test_mae.update(fake_targets, real_targets)
        self.test_pcc.update(fake_targets.double().flatten(), real_targets.double().flatten())
        self.test_psnr.update(fake_targets, real_targets)
        self.test_ssim.update(fake_targets, real_targets)

    def on_test_epoch_end(self):
        test_metrics = {
            'mae/test': self.test_mae.compute(),
            'pcc/test': self.test_pcc.compute(),
            'psnr/test': self.test_psnr.compute(),
            'ssim/test': self.test_ssim.compute(),
        }
        self.log_dict(test_metrics, on_step=False, on_epoch=True)
        self.test_mae.reset()
        self.test_pcc.reset()
        self.test_psnr.reset()
        self.test_ssim.reset()
