import lightning as L
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# from img2img.data.dataset import AlignedDataset
from img2img.data.dataset import DiffusionAlignedDataset
from img2img.networks.diffusion import define_model
from img2img.loss.diffusion import define_loss
# from img2img.utils.diffusion.noise_schedule import get_beta_schedule
from img2img.utils.diffusion.sampling import sample_image
from ema_pytorch import EMA

# LightningModule =================================================================
class Diffusion(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

# Create output directory =========================================================
        output_dir = Path(cfg['params']['output_dir'])
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

# Model ===========================================================================
        self.cfg = cfg
        self.model = define_model(cfg['model'])

# Loss ============================================================================
        self.criterion = define_loss(cfg)

# EMA =============================================================================
        if cfg['params']['ema']['active']:
            self.ema = EMA(self.model, beta=cfg['params']['ema']['rate'])
        else:
            self.ema = None

# Diffusion Beta Schedule =========================================================
        # self.num_timesteps = cfg['params']['diffusion']['num_timesteps']
        # betas = get_beta_schedule(
        #     beta_schedule=cfg['params']['diffusion']['beta_schedule'],
        #     beta_start=cfg['params']['diffusion']['beta_start'],
        #     beta_end=cfg['params']['diffusion']['beta_end'],
        #     num_diffusion_timesteps=self.num_timesteps
        # )
        # betas = torch.from_numpy(betas).float()
        # alphas = (1-betas).cumprod(dim=0)
        # self.register_buffer('alphas', alphas)

# Fast DDPM =======================================================================
        if cfg['params']['diffusion']['fast_ddpm']:
            print("Using Fast-DDPM")
            # skip = self.num_timesteps // cfg['params']['sampling']['timesteps']
            # print(f"Using Fast-DDPM with skip {skip} over {self.num_timesteps} timesteps")

# Fixed Initial Noise =============================================================
        # if cfg['params']['sampling']['fixed_initial_noise']:
        #     self.initial_noise = torch.randn(
        #         1,
        #         cfg['params']['sampling']['out_channels'],
        #         cfg['data']['image_size'],
        #         cfg['data']['image_size'],
        #         device=self.device
        #     )
        #     if cfg['params']['diffusion']['clip_noise']:
        #         self.initial_noise = self.initial_noise.clamp(-1, 1)
        #     print("Using fixed initial noise")
        # else:
        #     self.initial_noise = None

# Metrics =========================================================================
        # self.train_mae = MeanAbsoluteError()
        # self.train_pcc = PearsonCorrCoef()
        # self.train_psnr = PeakSignalNoiseRatio(data_range=2.0)
        # self.train_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

        self.val_mae = MeanAbsoluteError()
        self.val_pcc = PearsonCorrCoef()
        self.val_psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

        self.test_mae = MeanAbsoluteError()
        self.test_pcc = PearsonCorrCoef()
        self.test_psnr = PeakSignalNoiseRatio(data_range=2.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

# Dataset =========================================================================
    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = DiffusionAlignedDataset(
                cfg = self.cfg,
                input_dir=self.cfg['data']['train']['input_dir'],
                target_dir=self.cfg['data']['train']['target_dir'],
                image_size=self.cfg['data']['image_size'],
                ext=self.cfg['data']['ext'],
            )
            self.val_dataset = DiffusionAlignedDataset(
                cfg = self.cfg,
                input_dir=self.cfg['data']['val']['input_dir'],
                target_dir=self.cfg['data']['val']['target_dir'],
                image_size=self.cfg['data']['image_size'],
                ext=self.cfg['data']['ext'],
            )
        
        if stage == 'test':
            self.test_dataset = DiffusionAlignedDataset(
                cfg = self.cfg,
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
            optimizer = optim.Adam(self.model.parameters(), lr=args['lr'], betas=args['betas'])
        else:
            raise NotImplementedError(f"Optimizer {self.cfg['params']['optimizer']['name']} not implemented")

        return optimizer
    
# Number of parameters ============================================================
    def on_train_start(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        with open(Path(self.loggers[0].log_dir) / 'params.log', 'w') as file:
            file.write(f"Number of parameters model: {num_params}\n")

# Training Step ===================================================================
    def training_step(self, batch, batch_idx):
        # inputs, real_targets, _, _ = batch
        t, e, noisy_targets, cond_inputs, real_targets, _, _ = batch
        t = t.flatten().float()

# # Generate Noise ===============================================================
#         # (n, ch_out, h, w)
#         e = torch.randn_like(real_targets)
#         if self.cfg['params']['diffusion']['clip_noise']:
#             e = e.clamp(-1, 1)
#         n = e.shape[0]

# # Antithetic Sampling for Diffusion Timesteps ==================================
#         # (n,)
#         if self.fast_ddpm:
#             skip = self.num_timesteps // self.cfg['params']['sampling']['timesteps']
#             t_intervals = torch.arange(-1, self.num_timesteps, skip)
#             t_intervals[0] = 0

#             idx_1 = torch.randint(low=0, high=len(t_intervals), size=(n//2 + 1,))
#             idx_2 = len(t_intervals)-idx_1-1
#             idx = torch.cat([idx_1, idx_2], dim=0)[:n]
#             t = t_intervals[idx]
#         else:
#             t = torch.randint(
#                 low=0, high=self.num_timesteps, size=(n//2 + 1,)
#             )
#             t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

# # Add Noise to Real Target =====================================================
#         a = self.alphas.index_select(dim=0, index=t).view(-1, 1, 1, 1)
#         xt = real_targets*a.sqrt() + e*(1.0 - a).sqrt()

# Prediction & Loss ============================================================
        inputs = torch.cat((cond_inputs, noisy_targets), dim=1)
        if self.cfg['params']['pred'] == 'noise':
            # Predict Noise from Input and Noisy Target ========================
            pred_e = self.model(inputs, t)
            loss = self.criterion(
                true_noise=e, 
                pred_noise=pred_e
            )
        
        elif self.cfg['params']['pred'] == 'x0':
            # Predict x0 from Input and Noisy Target ===========================
            pred_x0 = self.model(inputs, t)
            loss = self.criterion(
                true_x0=real_targets,
                pred_x0=pred_x0
            )

# Log Loss ========================================================================
        # self.log_dict({'G_loss_step': loss}, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log_dict({'train/G_loss': loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))

# Log Metrics ====================================================================
        # fake_targets = fake_targets.detach()
        # real_targets = real_targets.detach()
        # fake_targets = torch.clamp(fake_targets, min=-1.0, max=1.0)
        # real_targets = torch.clamp(real_targets, min=-1.0, max=1.0)
        # self.train_mae.update(fake_targets, real_targets)
        # self.train_pcc.update(fake_targets.double().flatten(), real_targets.double().flatten())
        # self.train_psnr.update(fake_targets, real_targets)
        # self.train_ssim.update(fake_targets, real_targets)

        return loss

#==================================================================================
    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update()

#==================================================================================
    def on_train_epoch_end(self):
        global_step = self.global_step // 2  # self.global_step is the number of optimizer steps taken, in this case 2 steps per iteration because of the multi-optimizer training
        # train_metrics = {
        #     'mae/train': self.train_mae.compute(),
        #     'pcc/train': self.train_pcc.compute(),
        #     'psnr/train': self.train_psnr.compute(),
        #     'ssim/train': self.train_ssim.compute(),
        # }
        # self.log_dict(train_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # self.train_mae.reset()
        # self.train_pcc.reset()
        # self.train_psnr.reset()
        # self.train_ssim.reset()

# Save Image ======================================================================
        if self.current_epoch % self.cfg['params']['save_img_per_epoch'] == 0:
            output_image_train_dir = Path(self.loggers[0].log_dir)/"images"/"train"
            output_image_val_dir = Path(self.loggers[0].log_dir)/"images"/"val"
            if not output_image_train_dir.exists(): output_image_train_dir.mkdir(parents=True, exist_ok=True)
            if not output_image_val_dir.exists(): output_image_val_dir.mkdir(parents=True, exist_ok=True)
            self.model.eval()
            with torch.no_grad():
                for _, _, _, inputs, real_targets, _, _ in self.train_dataloader():
                    inputs = inputs[0].unsqueeze(0).to(self.device)
                    real_targets = real_targets[0].unsqueeze(0)
                    fake_targets = sample_image(
                        config=self.cfg,
                        model=self.model,
                        input_image=inputs,
                        initial_noise=None,
                        device=self.device,
                        create_list=False
                    )
                    self.train_dataset.dataset.save_image(fake_targets[0], output_image_train_dir/f"{self.current_epoch}_{global_step-1}_fake.png")
                    self.train_dataset.dataset.save_image(real_targets[0], output_image_train_dir/f"{self.current_epoch}_{global_step-1}_real.png")
                    train_fig = self.train_dataset.dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                    self.logger.experiment.add_figure('train/fig', train_fig, global_step-1)
                    break
                for _, _, _, inputs, real_targets, _, _ in self.val_dataloader():
                    inputs = inputs[0].unsqueeze(0).to(self.device)
                    real_targets = real_targets[0].unsqueeze(0)
                    fake_targets = sample_image(
                        config=self.cfg,
                        model=self.model,
                        input_image=inputs,
                        initial_noise=None,
                        device=self.device,
                        create_list=False
                    )
                    self.val_dataset.dataset.save_image(fake_targets[0], output_image_val_dir/f"{self.current_epoch}_{global_step-1}_fake.png")
                    self.val_dataset.dataset.save_image(real_targets[0], output_image_val_dir/f"{self.current_epoch}_{global_step-1}_real.png")
                    val_fig = self.val_dataset.dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                    self.logger.experiment.add_figure('val/fig', val_fig, global_step-1)
                    break
            self.model.train()

# Validation Step ==================================================================
    def validation_step(self, batch, batch_idx):
        t, e, noisy_targets, cond_inputs, real_targets, _, _ = batch
        t = t.flatten().float()
        inputs = torch.cat((cond_inputs, noisy_targets), dim=1)
        if self.cfg['params']['pred'] == 'noise':
            # Predict Noise from Input and Noisy Target ========================
            pred_e = self.model(inputs, t)
            loss = self.criterion(
                true_noise=e, 
                pred_noise=pred_e
            )
        
        elif self.cfg['params']['pred'] == 'x0':
            # Predict x0 from Input and Noisy Target ===========================
            pred_x0 = self.model(inputs, t)
            loss = self.criterion(
                true_x0=real_targets,
                pred_x0=pred_x0
            )
        
        self.log_dict({'val/G_loss': loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=inputs.size(0))

        # if self.current_epoch % self.cfg['params']['val_metric_per_epoch'] == 0:
        fake_targets = sample_image(
            config=self.cfg,
            model=self.model,
            input_image=cond_inputs,
            initial_noise=None,
            device=self.device,
            create_list=False
        )

        fake_targets = fake_targets.detach()
        real_targets = real_targets.detach()
        fake_targets = torch.clamp(fake_targets, min=-1.0, max=1.0)
        real_targets = torch.clamp(real_targets, min=-1.0, max=1.0)

        self.val_mae.update(fake_targets, real_targets)
        self.val_pcc.update(fake_targets.double().flatten(), real_targets.double().flatten())
        self.val_psnr.update(fake_targets, real_targets)
        self.val_ssim.update(fake_targets, real_targets)
        
    def on_validation_epoch_end(self):
        # if self.current_epoch % self.cfg['params']['val_metric_per_epoch'] == 0:
        val_metrics = {
            'val/mae': self.val_mae.compute(),
            'val/pcc': self.val_pcc.compute(),
            'val/psnr': self.val_psnr.compute(),
            'val/ssim': self.val_ssim.compute(),
        }
        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_mae.reset()
        self.val_pcc.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()

# Test Step =======================================================================
    def test_step(self, batch, batch_idx):
        _, _, _, inputs, real_targets, _, _ = batch
        fake_targets = sample_image(
            config=self.cfg,
            model=self.model,
            input_image=inputs,
            initial_noise=None,
            device=self.device,
            create_list=False
        )

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
