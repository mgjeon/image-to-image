# Import ==========================================================================
import os
import argparse
from pathlib import Path
# from time import perf_counter
import yaml
import torch
from setproctitle import setproctitle

from torch.utils.data import DataLoader

from img2img.networks.gan import define_G
from img2img.networks.diffusion import define_model
from img2img.data.dataset import AlignedDataset
from img2img.evaluation import get_last_epoch, get_last_version
from img2img.evaluation import calculate_metrics

# Main =========================================================================
if __name__ == "__main__":
# Load config ==================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--version', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--ckpt', type=str, choices=['best', 'last', 'epoch'], default='best')
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--save_meta', action='store_true', default=True)
    parser.add_argument('--use_sunpy_map', action='store_true', default=True,
                        help="Metrics within/outside the solar disk are calculated using sunpy.map")
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--timesteps', type=int, default=-1)
    parser.add_argument('--ema', action='store_true', default=False)
    args = parser.parse_args()
    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    setproctitle(cfg["name"])
# Find checkpoint ==============================================================
    output_dir = Path(cfg['params']['output_dir']) if args.output_dir is None else Path(args.output_dir)
    log_root = output_dir / "logs"
    version = get_last_version(log_root) if args.version == -1 else args.version
    log_dir = log_root / f"version_{version}"
    ckpt_dir = log_dir / "checkpoints"
    if args.ckpt_name is not None:
        ckpt_name = args.ckpt_name
    else:
        if args.ckpt == 'best':
            ckpt_name = sorted(ckpt_dir.glob('best*.ckpt'))[0].name
        elif args.ckpt == 'last':
            ckpt_name = sorted(ckpt_dir.glob('last.ckpt'))[0].name
        elif args.ckpt == 'epoch':
            epoch = get_last_epoch(ckpt_dir) if args.epoch == -1 else args.epoch
            ckpt_name = f"epoch={epoch}.ckpt"
    ckpt_path = ckpt_dir / ckpt_name
    print("Using checkpoint:", ckpt_path)

# Model ========================================================================
    # if cfg['model']['name'] == 'gan':
    #     from img2img.models import GAN
    #     model = GAN(cfg)
    # else:
    #     raise NotImplementedError(f"Model {cfg['model']['name']} not implemented")
    
# Trainer ======================================================================
    # import lightning as L
    # trainer = L.Trainer(
    #     accelerator=args.accelerator,
    #     devices=args.device,
    # )

# Test =========================================================================
    # trainer.test(model, ckpt_path=ckpt_path)

# Set device ===================================================================
    if args.device == -1:
        print("Using device: cpu")
        device = torch.device("cpu")
    else:
        print(f"Using device: {args.device}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model ===================================================================
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    eval_name = str(args.ckpt) + "_" + str(ckpt['epoch'])
    if cfg['model']['name'] == 'gan':
        model = define_G(cfg['model'])
        if args.ema:
            model_weights = {k.replace('ema.ema_model.', ''): v for k, v in ckpt["state_dict"].items() if k.startswith("ema.ema_model.")}
        else:
            model_weights = {k.replace("net_G.",''): v for k, v in ckpt["state_dict"].items() if k.startswith("net_G.")}
    elif cfg['model']['name'] == 'diffusion':
        model = define_model(cfg['model'])
        if args.ema:
            model_weights = {k.replace('ema.ema_model.', ''): v for k, v in ckpt["state_dict"].items() if k.startswith("ema.ema_model.")}
        else:
            model_weights = {k.replace("model.model.","model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.model.")}
    model.load_state_dict(model_weights)
    model = model.to(device)

    dataset_root = args.dataset_root if args.dataset_root is not None else cfg['data']['dataset_root']

    test_dataset = AlignedDataset(
        dataset_root = dataset_root,
        input_dir=cfg['data']['test']['input_dir'], 
        target_dir=cfg['data']['test']['target_dir'],
        image_size=cfg['data']['image_size'],
        ext=cfg['data']['ext'],
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['data']['test']['batch_size'], 
        shuffle=cfg['data']['test']['shuffle'],
        num_workers=cfg['data']['test']['num_workers'],
        pin_memory=cfg['data']['test']['pin_memory'],
        drop_last=cfg['data']['test']['drop_last']
    )


    if args.timesteps != -1:
        args.return_seq = True
        eval_name += f"_timesteps={args.timesteps}"

    if args.ema:
        out_dir = log_dir / "eval_ema" / eval_name / "out"
        csv_dir = log_dir / "eval_ema" / eval_name / "csv"
    else:
        out_dir = log_dir / "eval" / eval_name / "out"
        csv_dir = log_dir / "eval" / eval_name / "csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)


    test_metrics = calculate_metrics(
        model=model,
        cfg=cfg,
        dataset=test_dataset,
        loader=test_loader,
        device=device,
        out_dir=out_dir,
        csv_dir=csv_dir,
        args=args,
        stage='test',
        ckpt_path=ckpt_path,
    )
    print(test_metrics)
    