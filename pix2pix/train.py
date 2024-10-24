# Import =======================================================================
import logging
from pathlib import Path
from datetime import datetime
from time import perf_counter
import yaml
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lightning as L

from pipeline import AlignedDataset
from networks import define_G, define_D


# Create timestamped output folder ============================================
# def create_timestamped_folder(output_dir, resume=False):
#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         output_dir.mkdir(parents=True)

#     # Check existing folders that follow the pattern "YYYYMMDD_HHMMSS"
#     timestamp_folders = [f for f in output_dir.iterdir() if f.is_dir() and len(f.name) == 15 and f.name[:8].isdigit() and f.name[9:].isdigit()]
    
#     if timestamp_folders:
#         # Sort folders by timestamp (latest first)
#         timestamp_folders.sort(key=lambda x: x.name, reverse=True)
        
#         if resume:
#             return timestamp_folders[0]  # Return the latest timestamped folder if resume is True
#         else:
#             # Create new folder with current timestamp
#             now = datetime.now().strftime("%Y%m%d_%H%M%S")
#             new_timestamp_folder = output_dir / now
#     else:
#         # Create first timestamped folder if none exists
#         now = datetime.now().strftime("%Y%m%d_%H%M%S")
#         new_timestamp_folder = output_dir / now
    
#     new_timestamp_folder.mkdir()
#     return new_timestamp_folder


# Main =========================================================================
if __name__ == "__main__":
# Load config ==================================================================
    with open("configs/default.yaml") as file:
        cfg = yaml.safe_load(file)
        data = cfg['data']
        params = cfg['params']

# Create output directory ======================================================
        if params['resume_from'] is not None:
            output_dir = Path(params['output_dir_resume_from'])
        else:
            # output_dir = create_timestamped_folder(params['output_dir'], resume=params['resume'])
            output_dir = Path(params['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

# Save config ==================================================================
    with open(output_dir / "config.yaml", "w") as file:
        yaml.dump(cfg, file)

# Tensorboard ==================================================================
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

# Logging ======================================================================
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(output_dir / "log.log"))

# Seed =========================================================================
    L.seed_everything(params['seed'])
    torch.set_float32_matmul_precision(params['float32_matmul_precision'])

# Model, Optimizer, Loss, Dataset, DataLoader ==================================
    device = torch.device(params['device'])

    net_G = define_G(cfg).to(device)
    net_D = define_D(cfg).to(device)

    if params['optimizer']['name'] == "Adam":
        args = params['optimizer']['args']
        optim_G = optim.Adam(net_G.parameters(), lr=args['lr'], betas=args['betas'])
        optim_D = optim.Adam(net_D.parameters(), lr=args['lr'], betas=args['betas'])
    else:
        raise NotImplementedError("Optimizer not implemented")

    if params['loss']['generator']['name'] == 'L1Loss':
        criterion_G = torch.nn.L1Loss()
        lambda_factor = params['loss']['generator']['lambda']
    else:
        raise NotImplementedError("Loss (G) not implemented")
    
    if params['loss']['discriminator']['name'] == 'BCEWithLogitsLoss':
        criterion_D = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError("Loss (D) not implemented")
    
    train_dataset = AlignedDataset(
        data['train']['input_dir'], 
        data['train']['target_dir'],
        data['ext']
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=data['train']['batch_size'], 
        shuffle=True,
        num_workers=data['train']['num_workers'],
        drop_last=True
    )

    val_dataset = AlignedDataset(
        data['val']['input_dir'], 
        data['val']['target_dir'],
        data['ext']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=data['val']['batch_size'], 
        shuffle=False,
        num_workers=data['val']['num_workers'],
        drop_last=False
    )

# Setup Training ===============================================================
    output_image_train_dir = output_dir / "images/train"
    output_image_val_dir = output_dir / "images/val"
    output_model_dir = output_dir / "models"

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

    elif params['resume'] and (output_model_dir/"latest_checkpoint.pth").exists():
        checkpoint = torch.load(output_model_dir/"latest_checkpoint.pth")
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
        print(f"Resuming from latest checkpoint: {output_model_dir/'latest_checkpoint.pth'}")
        
    else:
        start_epoch = 1
        start_iteration = 0
        output_image_train_dir.mkdir(parents=True, exist_ok=True)
        output_image_val_dir.mkdir(parents=True, exist_ok=True)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        print("Starting from scratch")

# Start Training ===============================================================
    logger.info(f"Output directory: {output_dir.resolve()}")
    start_time = perf_counter()

    iteration = start_iteration
    for epoch in range(start_epoch, params['num_epochs']+1):
        D_losses = []
        G_losses = []
        train_loop = tqdm(train_loader, leave=True)
        train_loop.set_description(f"Epoch {epoch}/{params['num_epochs']}")
        for i, (inputs, real_target) in enumerate(train_loop):
            # print(i)
            # print(inputs.shape)
            # print(targets.shape)
            # import sys
            # sys.exit()

            inputs = inputs.to(device)
            real_target = real_target.to(device)

# Train Discriminator ==========================================================
            fake_target = net_G(inputs)
            D_real = net_D(inputs, real_target)
            D_real_loss = criterion_D(D_real, torch.ones_like(D_real))
            D_fake = net_D(inputs, fake_target.detach())
            D_fake_loss = criterion_D(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            optim_D.zero_grad()
            D_loss.backward()
            optim_D.step()

# Train Generator ==============================================================
            D_fake = net_D(inputs, fake_target)
            G_fake_loss = criterion_D(D_fake, torch.ones_like(D_fake))
            G_loss = G_fake_loss + lambda_factor*(criterion_G(fake_target, real_target)) 

            optim_G.zero_grad()
            G_loss.backward()
            optim_G.step()

# Update Iteration =============================================================
            iteration += 1

# Log Losses ===================================================================
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            if iteration % params['log_per_iteration'] == 0:
                train_loop.set_postfix(
                    D_loss = D_loss.item(),
                    G_loss = G_loss.item(),
                )
                writer.add_scalar("D_loss", D_loss.item(), iteration)
                writer.add_scalar("G_loss", G_loss.item(), iteration)
                
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
        torch.save(state, output_model_dir/"latest_checkpoint.pth")

# Log to file ==================================================================
        D_loss_avg = sum(D_losses) / len(D_losses)
        G_loss_avg = sum(G_losses) / len(G_losses)
        logger.info(f"Epoch {epoch}/{params['num_epochs']}: D_loss={D_loss_avg}, G_loss={G_loss_avg}")
        
# Save images ==================================================================
        if epoch % params['save_img_per_epoch'] == 0 or epoch == 1:
            net_G.eval()
            net_D.eval()
            with torch.no_grad():
                train_dataset.save_image(fake_target[0], output_image_train_dir/f"{epoch}_{iteration}_fake.png")
                train_dataset.save_image(real_target[0], output_image_train_dir/f"{epoch}_{iteration}_real.png")
                train_fig = train_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                writer.add_figure("train", train_fig, iteration)
                for inputs, real_target in val_loader:
                    inputs = inputs.to(device)
                    fake_target = net_G(inputs)
                    val_dataset.save_image(fake_target[0], output_image_val_dir/f"{epoch}_{iteration}_fake.png")
                    val_dataset.save_image(real_target[0], output_image_val_dir/f"{epoch}_{iteration}_real.png")
                    break
                val_fig = val_dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                writer.add_figure("val", val_fig, iteration)
            net_G.train()
            net_D.train()

# Save model ===================================================================
        if epoch % params['save_state_per_epoch'] == 0 or epoch == 1:
            torch.save(state, output_model_dir/f"{epoch}_{iteration}_checkpoint.pth")
            torch.save(net_G.state_dict(), output_model_dir/f"{epoch}_{iteration}_G.pth")
            # torch.save(net_D.state_dict(), output_model_dir/f"{epoch}_{iteration}_D.pth")

# End Training ================================================================
    end_time = perf_counter()
    logger.info(f"Training took {end_time-start_time} seconds")