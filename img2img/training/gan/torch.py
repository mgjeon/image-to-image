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
from torchmetrics import MeanAbsoluteError
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from img2img.dataset import AlignedDataset
from img2img.networks.gan import define_G, define_D
from img2img.utils.ema import EMAHelper


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

# Set device ====================================================================
    assert isinstance(params['devices'], list)
    devices = ",".join(map(str, params['devices'])) if len(params['devices']) > 1 else str(params['devices'][0])
    print(f"Using devices: {devices}")
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    device = torch.device(params['accelerator'])

# Model ========================================================================
    net_G = define_G(model_cfg)
    net_D = define_D(model_cfg)
    logger.info(f"# of parameters of Generator    : {sum([p.numel() for p in net_G.parameters()])}")
    logger.info(f"# of parameters of Discriminator: {sum([p.numel() for p in net_D.parameters()])}")
    if torch.cuda.device_count() > 1:
        net_G = torch.nn.DataParallel(net_G)
        net_D = torch.nn.DataParallel(net_D)
        cfg['params']['strategy'] = "dp"
    net_G = net_G.to(device)
    net_D = net_D.to(device)

# Optimizer ====================================================================
    if params['optimizer']['name'] == "Adam":
        args = params['optimizer']['args']
        optim_G = optim.Adam(net_G.parameters(), lr=args['lr'], betas=args['betas'])
        optim_D = optim.Adam(net_D.parameters(), lr=args['lr'], betas=args['betas'])
    else:
        raise NotImplementedError("Optimizer not implemented")

# Loss =========================================================================
    if cfg['params']['loss']['name'] == 'pix2pix':
        from img2img.loss.pix2pix import Loss
    elif cfg['params']['loss']['name'] == 'pix2pixhd':
        from img2img.loss.pix2pixhd import Loss
    elif cfg['params']['loss']['name'] == 'pix2pixcc':
        from img2img.loss.pix2pixcc import Loss
    criterion = Loss(cfg)

# Setup Training ===============================================================
    if params['resume_from'] is not None:
        checkpoint = torch.load(params['resume_from'])
        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        if start_epoch > params['num_epochs']:
            raise ValueError(f"Resuming epoch {start_epoch} is greater than num_epochs {params['num_epochs']}")
        net_G.load_state_dict(checkpoint["net_G"])
        net_D.load_state_dict(checkpoint["net_D"])
        net_G = net_G.to(device)
        net_D = net_D.to(device)
        optim_G.load_state_dict(checkpoint['optim_G'])
        optim_D.load_state_dict(checkpoint['optim_D'])
        print(f"Resuming from checkpoint: {params['resume_from']}")

    else:
        start_epoch = 0
        start_iteration = 0
        print("Starting from scratch")

# Model EMA ====================================================================
    if params['ema']['active']:
        ema_helper = EMAHelper(mu=params['ema']['rate'])
        ema_helper.register(net_G)
    else:
        ema_helper = None

# Metrics ======================================================================
    mae = MeanAbsoluteError().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    pearson = PearsonCorrCoef().to(device)

# Start Training ===============================================================
    logger.info(f"Output directory: {output_dir.resolve()}")
    start_time = perf_counter()

    iter_per_epoch = len(train_loader)
   
    iteration = start_iteration
    for epoch in range(start_epoch, params['num_epochs']):
        losses_D = []
        losses_G = []
        train_loop = tqdm(train_loader, leave=True)
        train_loop.set_description(f"Epoch {epoch}")
        for batch_idx, (inputs, real_targets, _, _) in enumerate(train_loop):
            # print(batch_idx)
            # print(inputs.shape)
            # print(real_target.shape)
            # import sys
            # sys.exit()

            inputs = inputs.to(device)
            real_targets = real_targets.to(device)

            loss_G, loss_D = criterion(net_G, net_D, inputs, real_targets)

# Train Generator ============================================================== 
            optim_G.zero_grad()
            loss_G.backward()

            if params['grad_clip']['clip']:
                torch.nn.utils.clip_grad_norm_(
                    parameters=net_G.parameters(),
                    max_norm=params['grad_clip']['max_norm']
                )

            optim_G.step()

# Update EMA ===================================================================
            if ema_helper is not None:
                ema_helper.update(net_G)

# Train Discriminator ==========================================================
            optim_D.zero_grad()
            loss_D.backward()

            if params['grad_clip']['clip']:
                torch.nn.utils.clip_grad_norm_(
                    parameters=net_D.parameters(),
                    max_norm=params['grad_clip']['max_norm']
                )

            optim_D.step()

# Log Loss =====================================================================
            losses_D.append(loss_D.item())
            losses_G.append(loss_G.item())

            if params['log_per_iteration'] == 1:
                log_condition = True
                epoch_log_condition = True
            else:
                log_condition = (iteration > 0) and ((iteration+1) % params['log_per_iteration'] == 0)
                epoch_log_condition = (iteration > 0) and (((iteration+1) % params['log_per_iteration'] == 0) or ((iteration+1) % iter_per_epoch == 0))
            if log_condition:
                writer.add_scalar("loss_D_step", loss_D.item(), iteration)
                writer.add_scalar("loss_G_step", loss_G.item(), iteration)
            if epoch_log_condition:
                writer.add_scalar("epoch", epoch, iteration)

# Update Iteration ============================================================
            iteration += 1

# Save latest checkpoint ======================================================
        state = {
                'net_G': net_G.state_dict(),
                'net_D': net_D.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'epoch': epoch,
                'iteration': iteration-1,
                'hparams': cfg
        }
        torch.save(state, output_model_dir/"last.ckpt")

# Log to tensorboard, file ====================================================
        loss_D_avg = sum(losses_D) / len(losses_D)
        loss_G_avg = sum(losses_G) / len(losses_G)
        writer.add_scalar("loss_D_epoch", loss_D_avg, iteration-1)
        writer.add_scalar("loss_G_epoch", loss_G_avg, iteration-1)
        logger.info(f"Epoch {epoch}: loss_D={loss_D_avg}, loss_G={loss_G_avg}")
        
# Save images ==================================================================
        if epoch % params['save_img_per_epoch'] == 0 or epoch == params['num_epochs'] - 1:
            net_G.eval()
            with torch.no_grad():
                for inputs, real_targets, _, _ in train_loader:
                    inputs = inputs[0].unsqueeze(0).to(device)
                    real_targets = real_targets[0].unsqueeze(0)
                    fake_targets = net_G(inputs)
                    real_targets = real_targets.to(device)
                    fake_targets = fake_targets.to(device)

                    train_dataset.save_image(fake_targets[0], output_image_train_dir/f"{epoch}_{iteration-1}_fake.png")
                    train_dataset.save_image(real_targets[0], output_image_train_dir/f"{epoch}_{iteration-1}_real.png")
                    train_fig = train_dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                    writer.add_figure("train/image", train_fig, iteration-1)

                    mae_value = mae(fake_targets, real_targets)
                    pixel_to_pixel_cc = pearson(fake_targets.flatten(), real_targets.flatten())
                    psnr_value = psnr(fake_targets, real_targets)
                    ssim_value = ssim(fake_targets, real_targets)
                    writer.add_scalar("train/mae", mae_value.item(), iteration-1)
                    writer.add_scalar("train/cc", pixel_to_pixel_cc.item(), iteration-1)
                    writer.add_scalar("train/psnr", psnr_value.item(), iteration-1)
                    writer.add_scalar("train/ssim", ssim_value.item(), iteration-1)

                    mae.reset()
                    pearson.reset()
                    psnr.reset()
                    ssim.reset()
                    break
                for inputs, real_targets, _, _ in val_loader:
                    inputs = inputs[0].unsqueeze(0).to(device)
                    real_targets = real_targets[0].unsqueeze(0)
                    fake_targets = net_G(inputs)
                    real_targets = real_targets.to(device)
                    fake_targets = fake_targets.to(device)

                    val_dataset.save_image(fake_targets[0], output_image_val_dir/f"{epoch}_{iteration-1}_fake.png")
                    val_dataset.save_image(real_targets[0], output_image_val_dir/f"{epoch}_{iteration-1}_real.png")
                    val_fig = val_dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                    writer.add_figure("val/image", val_fig, iteration-1)

                    mae_value = mae(fake_targets, real_targets)
                    pixel_to_pixel_cc = pearson(fake_targets.flatten(), real_targets.flatten())
                    psnr_value = psnr(fake_targets, real_targets)
                    ssim_value = ssim(fake_targets, real_targets)
                    writer.add_scalar("val/mae", mae_value.item(), iteration-1)
                    writer.add_scalar("val/cc", pixel_to_pixel_cc.item(), iteration-1)
                    writer.add_scalar("val/psnr", psnr_value.item(), iteration-1)
                    writer.add_scalar("val/ssim", ssim_value.item(), iteration-1)

                    mae.reset()
                    pearson.reset()
                    psnr.reset()
                    ssim.reset()
                    break
                
            net_G.train()

# Save model ===================================================================
        if epoch % params['save_state_per_epoch'] == 0 or epoch == params['num_epochs'] - 1:
            # torch.save(state, output_model_dir/f"{epoch}_{iteration-1}_checkpoint.pth")
            try:
                torch.save(net_G.module.state_dict(), output_model_dir/f"{epoch}_{iteration-1}_G.pth")
                torch.save(ema_helper.module.state_dict(), output_model_dir/f"{epoch}_{iteration-1}_G_ema.pth")
            except:
                torch.save(net_G.state_dict(), output_model_dir/f"{epoch}_{iteration-1}_G.pth")
                torch.save(ema_helper.state_dict(), output_model_dir/f"{epoch}_{iteration-1}_G_ema.pth")

# End Training ================================================================
    end_time = perf_counter()
    logger.info(f"Training time: {end_time-start_time} seconds")
    logger.info("PyTorch")