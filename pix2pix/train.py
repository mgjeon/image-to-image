# Import =======================================================================
import os
import logging
import argparse
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
import yaml
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pipeline import AlignedDataset
from networks import define_G, define_D
from loss import Loss


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
    args = parser.parse_args()
    with open(args.config) as file:
        cfg = yaml.safe_load(file)
        data = cfg['data']
        params = cfg['params']
        if args.resume_from is not None:
            params['resume_from'] = args.resume_from

# Set device ====================================================================
    assert isinstance(params['devices'], list)
    devices = ",".join(map(str, params['devices'])) if len(params['devices']) > 1 else str(params['devices'][0])
    print(f"Using devices: {devices}")
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    device = torch.device(params['accelerator'])

# Create output directory ======================================================
    output_dir = Path(params['output_dir'])
    log_root = output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    version = get_next_version(log_root)
    log_dir = log_root / f"version_{version}"
    log_dir.mkdir(parents=True, exist_ok=True)

# Tensorboard ==================================================================
    writer = SummaryWriter(log_dir)

# Logging ======================================================================
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_dir / "log.log"))

# Seed =========================================================================
    torch.manual_seed(params['seed'])
    torch.set_float32_matmul_precision(params['float32_matmul_precision'])

# Model ========================================================================
    net_G = define_G(cfg)
    net_D = define_D(cfg)
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
    criterion = Loss(cfg)
    
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

# Save config ==================================================================
    with open(log_dir / "hparams.yaml", "w") as file:
        yaml.dump(cfg, file)

# Setup Training ===============================================================
    output_image_dir = output_dir / "images" / f"version_{version}"
    output_image_train_dir = output_image_dir / "train"
    output_image_val_dir = output_image_dir / "val"
    output_image_train_dir.mkdir(parents=True, exist_ok=True)
    output_image_val_dir.mkdir(parents=True, exist_ok=True)

    output_model_dir = log_dir / "checkpoints"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    if params['resume_from'] is not None:
        checkpoint = torch.load(params['resume_from'])
        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration']
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

# Start Training ===============================================================
    logger.info(f"Output directory: {output_dir.resolve()}")
    start_time = perf_counter()

    iteration = start_iteration
    for epoch in range(start_epoch, params['num_epochs']):
        D_losses = []
        G_losses = []
        train_loop = tqdm(train_loader, leave=True)
        train_loop.set_description(f"Epoch {epoch}")
        for batch_idx, (inputs, real_targets) in enumerate(train_loop):
            # print(batch_idx)
            # print(inputs.shape)
            # print(real_target.shape)
            # import sys
            # sys.exit()

            inputs = inputs.to(device)
            real_targets = real_targets.to(device)

            loss_G, loss_D, fake_targets = criterion(net_G, net_D, inputs, real_targets)

# Train Generator ============================================================== 
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

# Train Discriminator ==========================================================
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

# Log Loss =====================================================================
            D_losses.append(loss_D.item())
            G_losses.append(loss_G.item())

            if iteration % params['log_per_iteration'] == 0:
                train_loop.set_postfix(
                    D_loss = loss_D.item(),
                    G_loss = loss_G.item(),
                )
                writer.add_scalar("D_loss_step", loss_D.item(), iteration)
                writer.add_scalar("G_loss_step", loss_G.item(), iteration)

# Update Iteration ============================================================
            iteration += 1
                
# Save latest checkpoint ======================================================
        state = {
                'net_G': net_G.state_dict(),
                'net_D': net_D.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'epoch': epoch,
                'iteration': iteration,
                'hparams': cfg
        }
        torch.save(state, output_model_dir/"last.ckpt")

# Log to tensorboard, file ====================================================
        D_loss_avg = sum(D_losses) / len(D_losses)
        G_loss_avg = sum(G_losses) / len(G_losses)
        writer.add_scalar("D_loss_epoch", D_loss_avg, iteration)
        writer.add_scalar("G_loss_epoch", G_loss_avg, iteration)
        logger.info(f"Epoch {epoch}: D_loss={D_loss_avg}, G_loss={G_loss_avg}")
        
# Save images ==================================================================
        if epoch % params['save_img_per_epoch'] == 0 or epoch == params['num_epochs'] - 1:
            net_G.eval()
            with torch.no_grad():
                train_dataset.save_image(fake_targets[0], output_image_train_dir/f"{epoch}_{iteration}_fake.png")
                train_dataset.save_image(real_targets[0], output_image_train_dir/f"{epoch}_{iteration}_real.png")
                train_fig = train_dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                writer.add_figure("train", train_fig, iteration)
                for inputs, real_targets in val_loader:
                    inputs = inputs.to(device)
                    fake_targets = net_G(inputs)
                    val_dataset.save_image(fake_targets[0], output_image_val_dir/f"{epoch}_{iteration}_fake.png")
                    val_dataset.save_image(real_targets[0], output_image_val_dir/f"{epoch}_{iteration}_real.png")
                    break
                val_fig = val_dataset.create_figure(inputs[0], real_targets[0], fake_targets[0])
                writer.add_figure("val", val_fig, iteration)
            net_G.train()

# Save model ===================================================================
        if epoch % params['save_state_per_epoch'] == 0 or epoch == params['num_epochs'] - 1:
            torch.save(state, output_model_dir/f"{epoch}_{iteration}_checkpoint.pth")
            try:
                torch.save(net_G.module.state_dict(), output_model_dir/f"{epoch}_{iteration}_G.pth")
            except:
                torch.save(net_G.state_dict(), output_model_dir/f"{epoch}_{iteration}_G.pth")

# End Training ================================================================
    end_time = perf_counter()
    logger.info(f"Training took {end_time-start_time} seconds")