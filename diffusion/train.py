# Import =======================================================================
import os
import logging
import argparse
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
from setproctitle import setproctitle
import yaml
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pipeline import AlignedDataset
from networks import define_model
from loss import define_loss
from ema import EMAHelper
from noise_schedule import get_beta_schedule
from sampling import sample_image


# Get next version =============================================================
def get_next_version(output_dir):
    existing_versions = [int(d.name.split("_")[-1]) for d in output_dir.glob("version_*")]
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1


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
        data = cfg['data']
        params = cfg['params']
        model_cfg = cfg['model']
        if args.resume_from is not None:
            cfg['params']['resume_from'] = args.resume_from
        if args.num_epochs is not None:
            cfg['params']['num_epochs'] = args.num_epochs

# Set process name =============================================================
    setproctitle(cfg["name"])

# Create output directory ======================================================
    output_dir = Path(params['output_dir'])
    log_root = output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    version = get_next_version(log_root)
    log_dir = log_root / f"version_{version}"
    log_dir.mkdir(parents=True, exist_ok=True)

    output_image_train_dir = log_dir / "images" / "train"
    output_image_val_dir = log_dir / "images" / "val"
    output_image_train_dir.mkdir(parents=True, exist_ok=True)
    output_image_val_dir.mkdir(parents=True, exist_ok=True)

# Tensorboard ==================================================================
    writer = SummaryWriter(log_dir)

# Logging ======================================================================
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_dir / "log.log"))

# Save config ==================================================================
    with open(log_dir / "hparams.yaml", "w") as file:
        yaml.dump(cfg, file)

    output_model_dir = log_dir / "checkpoints"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    with open(output_model_dir / "config.yaml", "w") as file:
        yaml.dump(model_cfg, file)

# Seed =========================================================================
    torch.manual_seed(params['seed'])
    torch.set_float32_matmul_precision(params['float32_matmul_precision'])

# Dataset, DataLoader ==========================================================
    train_dataset = AlignedDataset(
        input_dir=data['train']['input_dir'], 
        target_dir=data['train']['target_dir'],
        image_size=data['image_size'],
        ext=data['ext']
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=data['train']['batch_size'], 
        shuffle=data['train']['shuffle'],
        num_workers=data['train']['num_workers'],
        pin_memory=data['train']['pin_memory'],
        drop_last=data['train']['drop_last']
    )

    val_dataset = AlignedDataset(
        input_dir=data['val']['input_dir'], 
        target_dir=data['val']['target_dir'],
        image_size=data['image_size'],
        ext=data['ext']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=data['val']['batch_size'], 
        shuffle=data['val']['shuffle'],
        num_workers=data['val']['num_workers'],
        pin_memory=data['val']['pin_memory'],
        drop_last=data['val']['drop_last']
    )

# Set device ===================================================================
    assert isinstance(params['devices'], list)
    devices = ",".join(map(str, params['devices'])) if len(params['devices']) > 1 else str(params['devices'][0])
    print(f"Using devices: {devices}")
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    device = torch.device(params['accelerator'])

# Model ========================================================================
    model = define_model(model_cfg)
    logger.info(f"# of parameters of Model: {sum(p.numel() for p in model.parameters())}")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        cfg['params']['strategy'] = "dp"
    model = model.to(device)

# Optimizer ====================================================================
    if params['optimizer']['name'] == "Adam":
        args = params['optimizer']['args']
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=args['betas'])
    else:
        raise NotImplementedError("Optimizer not implemented")

# Loss =========================================================================
    criterion = define_loss(cfg)

# Resume from checkpoint =======================================================
    if params['resume_from'] is not None:
        checkpoint = torch.load(params['resume_from'], map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        if start_epoch > params['num_epochs']:
            raise ValueError(f"Resuming epoch {start_epoch} is greater than num_epochs {params['num_epochs']}")
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resuming from checkpoint: {params['resume_from']}")

    else:
        start_epoch = 0
        start_iteration = 0
        print("Starting from scratch")

# Model EMA ====================================================================
    if params['ema']['active']:
        ema_helper = EMAHelper(mu=params['ema']['rate'])
        ema_helper.register(model)
    else:
        ema_helper = None

# Diffusion Beta Schedule ======================================================
    num_timesteps = params['diffusion']['num_timesteps']
    betas = get_beta_schedule(
        beta_schedule=params['diffusion']['beta_schedule'],
        beta_start=params['diffusion']['beta_start'],
        beta_end=params['diffusion']['beta_end'],
        num_diffusion_timesteps=num_timesteps
    )
    betas = torch.from_numpy(betas).float().to(device)
    alphas = (1-betas).cumprod(dim=0)

# Fast DDPM ====================================================================
    fast_ddpm = params['diffusion']['fast_ddpm']
    if fast_ddpm:
        skip = num_timesteps // params['sampling']['timesteps']
        print(f"Using Fast-DDPM with skip {skip} over {num_timesteps} timesteps")

# Fixed Initial Noise ==========================================================
    try:
        out_channels = cfg['model']['args']['output_nc']
    except:
        out_channels = cfg['model']['args']['out_channels']
    if params['sampling']['fixed_initial_noise']:
        initial_noise = torch.randn(
            1,
            out_channels,
            data['image_size'],
            data['image_size'],
            device=device
        )
        print("Using fixed initial noise")
    else:
        initial_noise = None

# Start Training ===============================================================
    logger.info(f"Output directory: {output_dir.resolve()}")
    start_time = perf_counter()

    iter_per_epoch = len(train_loader)
   
    iteration = start_iteration
    for epoch in range(start_epoch, params['num_epochs']):
        losses = []
        train_loop = tqdm(train_loader, leave=True)
        train_loop.set_description(f"Epoch {epoch}")
        for batch_idx, (inputs, real_targets, _, _) in enumerate(train_loop):
            model.train()
            # print(batch_idx)
            # print(inputs.shape)
            # print(real_target.shape)
            # import sys
            # sys.exit()

# Input Image ==================================================================
            # (n, ch_in, h, w)
            inputs = inputs.to(device)

# Real Target Image ============================================================
            # (n, ch_out, h, w)
            real_targets = real_targets.to(device)

# Generate Noise ===============================================================
            # (n, ch_out, h, w)
            e = torch.randn_like(real_targets)
            n = e.shape[0]

# Antithetic Sampling for Diffusion Timesteps ==================================
            # (n,)
            if fast_ddpm:
                skip = num_timesteps // params['sampling']['timesteps']
                t_intervals = torch.arange(-1, num_timesteps, skip)
                t_intervals[0] = 0

                idx_1 = torch.randint(low=0, high=len(t_intervals), size=(n//2 + 1,))
                idx_2 = len(t_intervals)-idx_1-1
                idx = torch.cat([idx_1, idx_2], dim=0)[:n]
                t = t_intervals[idx].to(device)
            else:
                t = torch.randint(
                    low=0, high=num_timesteps, size=(n//2 + 1,)
                ).to(device)
                t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]

# Add Noise to Real Target =====================================================
            a = alphas.index_select(dim=0, index=t).view(-1, 1, 1, 1)
            xt = real_targets*a.sqrt() + e*(1.0 - a).sqrt()

# Prediction & Loss ============================================================
            if params['pred'] == 'noise':
# Predict Noise from Input and Noisy Target ===================================
                pred_e = model(torch.cat([inputs, xt], dim=1), t.float())
                loss = criterion(
                    true_noise=e, 
                    pred_noise=pred_e
                )
            
            elif params['pred'] == 'x0':
# Predict x0 from Input and Noisy Target ======================================
                pred_e = model(torch.cat([inputs, xt], dim=1), t.float())
                pred_x0 = (xt - pred_e*(1.0 - a).sqrt()) / a.sqrt()
                loss = criterion(
                    true_x0=real_targets, 
                    pred_x0=pred_x0
                )
            
            elif params['pred'] == 'both':
                pred_e = model(torch.cat([inputs, xt], dim=1), t.float())
                pred_x0 = (xt - pred_e*(1.0 - a).sqrt()) / a.sqrt()
                loss = criterion(
                    true_noise=e,
                    pred_noise=pred_e,
                    true_x0=real_targets,
                    pred_x0=pred_x0
                )
                
# Train Model ================================================================== 
            optimizer.zero_grad()
            loss.backward()

            if params['grad_clip']['clip']:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=params['grad_clip']['max_norm']
                )
        
            optimizer.step()

# Update EMA ===================================================================
            if ema_helper is not None:
                ema_helper.update(model)

# Log Loss =====================================================================
            losses.append(loss.item())

            if params['log_per_iteration'] == 1:
                log_condition = True
                epoch_log_condition = True
            else:
                log_condition = (iteration > 0) and ((iteration+1) % params['log_per_iteration'] == 0)
                epoch_log_condition = (iteration > 0) and (((iteration+1) % params['log_per_iteration'] == 0) or ((iteration+1) % iter_per_epoch == 0))
            if log_condition:
                writer.add_scalar("loss_step", loss.item(), iteration)
            if epoch_log_condition:
                writer.add_scalar("epoch", epoch, iteration)

# Update Iteration ============================================================
            iteration += 1

# Save latest checkpoint ======================================================
        state = {
                'model': model.state_dict(),
                'model_ema': ema_helper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'iteration': iteration-1,
                'hparams': cfg
        }
        torch.save(state, output_model_dir/"last.ckpt")

# Log to tensorboard, file ====================================================
        loss_avg = sum(losses) / len(losses)
        writer.add_scalar("loss_epoch", loss_avg, iteration-1)
        logger.info(f"Epoch {epoch}: loss={loss_avg}")
        
# Save images ==================================================================
        if epoch % params['save_img_per_epoch'] == 0 or epoch == params['num_epochs'] - 1:
            model.eval()
            with torch.no_grad():
                for inputs, real_targets, _, _ in train_loader:
                    inputs = inputs[0].unsqueeze(0).to(device)
                    real_targets = real_targets[0].unsqueeze(0).to(device)
                    fake_targets = sample_image(
                        config=cfg,
                        model=model,
                        input_image=inputs,
                        initial_noise=initial_noise,
                        device=device,
                        create_list=False
                    )
                    train_dataset.save_image(fake_targets[0], output_image_train_dir/f"{epoch}_{iteration-1}_fake.png")
                    train_dataset.save_image(real_targets[0], output_image_train_dir/f"{epoch}_{iteration-1}_real.png")
                    train_fig = train_dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                    writer.add_figure("train", train_fig, iteration-1)
                    break
                for inputs, real_targets, _, _ in val_loader:
                    inputs = inputs.to(device)
                    fake_targets = sample_image(
                        config=cfg,
                        model=model,
                        input_image=inputs,
                        initial_noise=initial_noise,
                        device=device,
                        create_list=False
                    )
                    val_dataset.save_image(fake_targets[0], output_image_val_dir/f"{epoch}_{iteration-1}_fake.png")
                    val_dataset.save_image(real_targets[0], output_image_val_dir/f"{epoch}_{iteration-1}_real.png")
                    break
                val_fig = val_dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                writer.add_figure("val", val_fig, iteration-1)
            model.train()

# Save model ===================================================================
        if epoch % params['save_state_per_epoch'] == 0 or epoch == params['num_epochs'] - 1:
            # torch.save(state, output_model_dir/f"{epoch}_{iteration-1}_checkpoint.pth")
            try:
                torch.save(model.module.state_dict(), output_model_dir/f"{epoch}_{iteration-1}.pth")
                torch.save(ema_helper.module.state_dict(), output_model_dir/f"{epoch}_{iteration-1}_ema.pth")
            except:
                torch.save(model.state_dict(), output_model_dir/f"{epoch}_{iteration-1}.pth")
                torch.save(ema_helper.state_dict(), output_model_dir/f"{epoch}_{iteration-1}_ema.pth")


# End Training ================================================================
    end_time = perf_counter()
    logger.info(f"Training time: {end_time-start_time} seconds")
    logger.info("PyTorch")