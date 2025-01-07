# Import ==========================================================================
import argparse
from pathlib import Path
from time import perf_counter
import yaml
import torch
from setproctitle import setproctitle

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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

    setproctitle(cfg["name"])
# Seed =========================================================================
    L.seed_everything(cfg['params']['seed'])
    torch.set_float32_matmul_precision(cfg['params']['float32_matmul_precision'])

# Model =========================================================================
    if cfg['model']['name'] == 'gan':
        from img2img.models import GAN
        model = GAN(cfg)
    elif cfg['model']['name'] == 'diffusion':
        from img2img.models import Diffusion
        model = Diffusion(cfg)
    else:
        raise NotImplementedError(f"Model {cfg['model']['name']} not implemented")

# Callback ======================================================================
    checkpoint_callback = ModelCheckpoint(
        filename='epoch={epoch}',
        auto_insert_metric_name=False,
        every_n_epochs=cfg['params']['save_state_per_epoch'],
        save_top_k=-1,
        save_last=True,
    )

    # checkpoint_callback_val_loss = ModelCheckpoint(
    #     filename='epoch={epoch}-val_loss={val/G_loss:.2g}',
    #     auto_insert_metric_name=False,
    #     monitor='val/G_loss',
    #     mode='min',
    #     save_top_k=1,
    #     save_last=False,
    # )

    # checkpoint_callback_mae = ModelCheckpoint(
    #     filename='epoch={epoch}-mae={val/mae:.2g}',
    #     auto_insert_metric_name=False,
    #     monitor='val/mae',
    #     mode='min',
    #     save_top_k=1,
    # )

    # checkpoint_callback_pcc = ModelCheckpoint(
    #     filename='epoch={epoch}-pcc={val/pcc:.2f}',
    #     auto_insert_metric_name=False,
    #     monitor='val/pcc',
    #     mode='max',
    #     save_top_k=1,
    # )

    # checkpoint_callback_psnr = ModelCheckpoint(
    #     filename='epoch={epoch}-psnr={val/psnr:.2f}',
    #     auto_insert_metric_name=False,
    #     monitor='val/psnr',
    #     mode='max',
    #     save_top_k=1,
    # )

    checkpoint_callback_ssim = ModelCheckpoint(
        filename='epoch={epoch}-ssim={val/ssim:.2f}',
        auto_insert_metric_name=False,
        monitor='val/ssim',
        mode='max',
        save_top_k=1,
    )

    tb_logger = TensorBoardLogger(
        save_dir=model.output_dir,
        name='logs',
        default_hp_metric=False,
    )

# Trainer =======================================================================
    if "ddp" in cfg['params']['strategy']:
        if len(cfg['params']['devices']) > 1 and cfg['model']['name'] == 'gan':
            strategy = "ddp_find_unused_parameters_true"
            cfg['params']['strategy'] = strategy
        else:
            strategy = "auto"
            cfg['params']['strategy'] = strategy
    else:
        strategy = cfg['params']['strategy']

    trainer = L.Trainer(
        default_root_dir=model.output_dir,
        callbacks=[
            checkpoint_callback, 
            # checkpoint_callback_val_loss,
            # checkpoint_callback_mae,
            # checkpoint_callback_pcc,
            # checkpoint_callback_psnr,
            checkpoint_callback_ssim,
        ],
        logger=[tb_logger],
        # log_every_n_steps=cfg['params']['log_per_iteration'],
        check_val_every_n_epoch=cfg['params']['val_per_epoch'],
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
    with open(Path(trainer.log_dir) / 'time.log', 'w') as file:
        file.write(f"Training time: {end_time-start_time} seconds")
        file.write("\nPyTorch Lightning")

# Test =========================================================================
    trainer.test(model, ckpt_path="best")