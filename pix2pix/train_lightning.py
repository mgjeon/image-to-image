# Import ==========================================================================
import argparse
from pathlib import Path
from time import perf_counter
import yaml
import torch
from torch import optim
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pipeline import AlignedDataset
from networks import define_G, define_D
from loss import Loss


# LightningModule =================================================================
class Pix2Pix(L.LightningModule):
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
        self.net_G = define_G(cfg)
        self.net_D = define_D(cfg)

# Loss ============================================================================
        self.criterion = Loss(cfg)
        
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

# DataLoader ======================================================================
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg['data']['train']['batch_size'],
            shuffle=self.cfg['data']['train']['shuffle'],
            num_workers=self.cfg['data']['train']['num_workers'],
            pin_memory=self.cfg['data']['train']['pin_memory'],
            drop_last=self.cfg['data']['train']['drop_last'],
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg['data']['val']['batch_size'],
            shuffle=self.cfg['data']['val']['shuffle'],
            num_workers=self.cfg['data']['val']['num_workers'],
            pin_memory=self.cfg['data']['val']['pin_memory'],
            drop_last=self.cfg['data']['val']['drop_last'],
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

# Training Step ===================================================================
    def training_step(self, batch, batch_idx):
        inputs, real_targets = batch

        optim_G, optim_D = self.optimizers()
        
        loss_G, loss_D = self.criterion(self.net_G, self.net_D, inputs, real_targets)

# Train Generator =================================================================
        optim_G.zero_grad()
        self.manual_backward(loss_G)
        optim_G.step()

# Train Discriminator =============================================================
        optim_D.zero_grad()
        self.manual_backward(loss_D)
        optim_D.step()

# Log Loss ========================================================================
        self.log('G_loss_step', loss_G, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('D_loss_step', loss_D, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('G_loss_epoch', loss_G, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('D_loss_epoch', loss_D, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def on_train_epoch_end(self):
        global_step = self.global_step // 2  # self.global_step is the number of optimizer steps taken, in this case 2 steps per iteration because of the multi-optimizer training

# Save Image ======================================================================
        if self.current_epoch % self.cfg['params']['save_img_per_epoch'] == 0:
            output_image_train_dir = Path(self.loggers[0].log_dir)/"images"/"train"
            output_image_val_dir = Path(self.loggers[0].log_dir)/"images"/"val"
            if not output_image_train_dir.exists(): output_image_train_dir.mkdir(parents=True, exist_ok=True)
            if not output_image_val_dir.exists(): output_image_val_dir.mkdir(parents=True, exist_ok=True)
            self.net_G.eval()
            with torch.no_grad():
                for inputs, real_target in self.train_dataloader():
                    inputs = inputs.to(self.device)
                    fake_target = self.net_G(inputs)
                    self.train_dataset.save_image(fake_target[0], output_image_train_dir/f"{self.current_epoch}_{global_step-1}_fake.png")
                    self.train_dataset.save_image(real_target[0], output_image_train_dir/f"{self.current_epoch}_{global_step-1}_real.png")
                    train_fig = self.train_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                    self.logger.experiment.add_figure('train', train_fig, global_step-1)
                    break
                for inputs, real_target in self.val_dataloader():
                    inputs = inputs.to(self.device)
                    fake_target = self.net_G(inputs)
                    self.val_dataset.save_image(fake_target[0], output_image_val_dir/f"{self.current_epoch}_{global_step-1}_fake.png")
                    self.val_dataset.save_image(real_target[0], output_image_val_dir/f"{self.current_epoch}_{global_step-1}_real.png")
                    val_fig = self.val_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                    self.logger.experiment.add_figure('val', val_fig, global_step-1)
                    break
            self.net_G.train()

# Save Model =======================================================================
        if self.current_epoch % self.cfg['params']['save_state_per_epoch'] == 0 or self.current_epoch == self.cfg['params']['num_epochs']-1:
            ckpt_path = Path(self.loggers[0].log_dir)/"checkpoints"
            if not ckpt_path.exists(): ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.net_G.state_dict(), ckpt_path/f"{self.current_epoch}_{global_step-1}_G.pth")


# Main =========================================================================
if __name__ == "__main__":
# Load config ==================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    args = parser.parse_args()
    with open(args.config) as file:
        cfg = yaml.safe_load(file)  
        if args.resume_from is not None:
            cfg['params']['resume_from'] = args.resume_from
        if args.num_epochs is not None:
            cfg['params']['num_epochs'] = args.num_epochs

# Seed =========================================================================
    L.seed_everything(cfg['params']['seed'])
    torch.set_float32_matmul_precision(cfg['params']['float32_matmul_precision'])

# Model =========================================================================
    model = Pix2Pix(cfg)

# Callback ======================================================================
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}',
        every_n_epochs=cfg['params']['save_state_per_epoch'],
        save_top_k=cfg['params']['save_top_k'],
        save_last=True
    )

    tb_logger = TensorBoardLogger(
        save_dir=model.output_dir,
        name='logs',
        default_hp_metric=False,
    )

# Trainer =======================================================================
    if "ddp" in cfg['params']['strategy']:
        if len(cfg['params']['devices']) > 1:
            strategy = "ddp_find_unused_parameters_true"
            cfg['params']['strategy'] = strategy
        else:
            strategy = "auto"
            cfg['params']['strategy'] = strategy
    else:
        strategy = cfg['params']['strategy']

    trainer = L.Trainer(
        default_root_dir=model.output_dir,
        callbacks=[checkpoint_callback],
        logger=[tb_logger],
        log_every_n_steps=cfg['params']['log_per_iteration'],
        max_epochs=cfg['params']['num_epochs'],
        accelerator=cfg['params']['accelerator'],
        devices=cfg['params']['devices'],
        precision=cfg['params']['precision'],
        strategy=strategy,
    )

# Training ======================================================================
    start_time = perf_counter()
    if cfg['params']['resume_from'] is not None:
        trainer.fit(model, ckpt_path=cfg['params']['resume_from'])
    else:
        trainer.fit(model)

# End Training ==================================================================
    end_time = perf_counter()
    with open(Path(trainer.log_dir) / 'log.log', 'w') as file:
        file.write(f"Training time: {end_time-start_time} seconds")
        file.write("\nPyTorch Lightning")