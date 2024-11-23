import os
import gc
import argparse
import logging
from pathlib import Path

import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import DataLoader

from torchmetrics import MeanAbsoluteError
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from networks import define_G
from pipeline import AlignedDataset

from time import perf_counter

# Get next version ===============================================================
def get_next_version(output_dir):
    existing_versions = [int(d.name.split("_")[-1]) for d in output_dir.glob("version_*")]
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1


# Metrics ========================================================================
def calculate_metrics(model, dataset, loader, device, log_dir, output_dir, args, stage="Validation"):
    binning = args.binning
    subsample = args.subsample
    model.eval()
    with torch.no_grad():
        mae = MeanAbsoluteError().to(device)                  # 0.0 is best
        psnr = PeakSignalNoiseRatio().to(device)              # +inf is best
        ssim = StructuralSimilarityIndexMeasure().to(device)  # 1.0 is best
        pearson = PearsonCorrCoef().to(device)                # 1.0 is best
        
        maes = []
        psnrs = []
        ssims = []
        pearsons = []

        avgpool2d = torch.nn.AvgPool2d(binning)
        maes_binning = []
        psnrs_binning = []
        ssims_binning = []
        pearsons_binning = []

        for i, (inputs, real_target, _, target_name) in enumerate(tqdm(loader, desc=stage)):
            inputs = inputs.to(device)
            real_target = real_target.to(device)
            fake_target = model(inputs)
            if i == 0:
                fig = dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                fig.savefig(log_dir / f"{stage}_example.png")
                print(inputs.shape)
                print(real_target.shape)
                print(fake_target.shape)

            bs = inputs.size(0)

            for i in range(bs):
                if args.save_meta:
                    target_file = Path(dataset.target_dir) / target_name[i]
                    target_file = str(target_file) + ".npz"
                    target_meta = np.load(target_file, allow_pickle=True)['metas']
                    np.savez(output_dir / f"{target_name[i]}_fake.npz", data=fake_target[i].cpu().numpy(), metas=target_meta)
                else:
                    dataset.save_image(fake_target[0], output_dir / f"{target_name[0]}_fake.png")
                    dataset.save_image(real_target[0], output_dir / f"{target_name[0]}_real.png")
                
            mae_value = mae(fake_target, real_target)
            pixel_to_pixel_cc = pearson(fake_target.flatten(), real_target.flatten())
            psnr_value = psnr(fake_target, real_target)
            ssim_value = ssim(fake_target, real_target)

            maes.append(mae_value.item())
            psnrs.append(psnr_value.item())
            ssims.append(ssim_value.item())
            pearsons.append(pixel_to_pixel_cc.item())

            mae.reset()
            pearson.reset()
            psnr.reset()
            ssim.reset()
            
            if args.binning > 1:
                inputs_binning = avgpool2d(inputs)
                real_target_binning = avgpool2d(real_target)
                fake_target_binning = avgpool2d(fake_target)
                if i == 0:
                    fig = dataset.create_figure(inputs[0], real_target_binning[0], fake_target_binning[0])
                    fig.savefig(log_dir / f"{stage}_example_binning.png")
                    print(inputs_binning.shape)
                    print(real_target_binning.shape)
                    print(fake_target_binning.shape)
                    
                mae_value_binning = mae(fake_target_binning, real_target_binning)
                pixel_to_pixel_cc_binning = pearson(fake_target_binning.flatten(), real_target_binning.flatten())
                psnr_value_binning = psnr(fake_target_binning, real_target_binning)
                ssim_value_binning = ssim(fake_target_binning, real_target_binning)

                maes_binning.append(mae_value_binning.item())
                psnrs_binning.append(psnr_value_binning.item())
                ssims_binning.append(ssim_value_binning.item())
                pearsons_binning.append(pixel_to_pixel_cc_binning.item())
                
                mae.reset()
                psnr.reset()
                ssim.reset()
                pearson.reset()

            del inputs
            del real_target
            del fake_target
            if args.binning > 1:
                del inputs_binning
                del real_target_binning
                del fake_target_binning

            gc.collect()
            torch.cuda.empty_cache()

            if subsample > 0:
                if (i+1) == subsample:
                    break

        mae = sum(maes) / len(maes)
        psnr = sum(psnrs) / len(psnrs)
        ssim = sum(ssims) / len(ssims)
        pearson = sum(pearsons) / len(pearsons)

        if args.binning > 1:
            mae_binning = sum(maes_binning) / len(maes_binning)
            psnr_binning = sum(psnrs_binning) / len(psnrs_binning)
            ssim_binning = sum(ssims_binning) / len(ssims_binning)
            pearson_binning = sum(pearsons_binning) / len(pearsons_binning)

            return {
                "MAE": mae,
                "PSNR": psnr,
                "SSIM": ssim,
                "Pearson": pearson,
                "MAE_binning": mae_binning,
                "PSNR_binning": psnr_binning,
                "SSIM_binning": ssim_binning,
                "Pearson_binning": pearson_binning
            }
        else:
            return {
                "MAE": mae,
                "PSNR": psnr,
                "SSIM": ssim,
                "Pearson": pearson
            }


# Main ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--binning", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=10)
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--save_meta", action="store_true")
    args = parser.parse_args()

    print(f"Using devices: {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.hparams) as file:
        cfg = yaml.safe_load(file)
        if cfg.get('cfg') is not None:
            cfg = cfg['cfg']  
        data = cfg['data']
        params = cfg['params']
    
    with open(args.config) as file:
        model_cfg = yaml.safe_load(file)

    G = define_G(model_cfg)
    model_pth = torch.load(args.model, map_location=device, weights_only=True)
    G.load_state_dict(model_pth)
    G = G.to(device)

    val_dataset = AlignedDataset(
        input_dir=data['val']['input_dir'], 
        target_dir=data['val']['target_dir'],
        image_size=data['image_size'],
        ext=data['ext'],
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=data['val']['batch_size'], 
        shuffle=data['val']['shuffle'],
        num_workers=data['val']['num_workers'],
        pin_memory=data['val']['pin_memory'],
        drop_last=data['val']['drop_last']
    )

    test_dataset = AlignedDataset(
        input_dir=data['test']['input_dir'], 
        target_dir=data['test']['target_dir'],
        image_size=data['image_size'],
        ext=data['ext'],
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=data['test']['batch_size'], 
        shuffle=data['test']['shuffle'],
        num_workers=data['test']['num_workers'],
        pin_memory=data['test']['pin_memory'],
        drop_last=data['test']['drop_last']
    )

    log_root = Path(args.model).parent.parent / "metrics" if args.log_root is None else Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    version = get_next_version(log_root)
    log_dir = log_root / f"version_{version}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_dir / "metric.log", "w"))

    with open(log_dir / "config.yaml", "w") as file:
        yaml.dump(args, file)

    val_dir = log_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    val_metrics = calculate_metrics(
        model=G,
        dataset=val_dataset,
        loader=val_loader,
        device=device,
        log_dir=log_dir,
        output_dir=val_dir,
        args=args,
        stage="Validation"
    )
    end_time = perf_counter()
    with open(log_dir / "val_metrics.yaml", "w") as file:
        yaml.dump(val_metrics, file)
    logger.info(f"Validation time: {end_time - start_time:.2f} s, total {len(val_dataset)} samples")
    logger.info(f"Validation MAE: {val_metrics['MAE']:.2f}")
    logger.info(f"Validation PSNR: {val_metrics['PSNR']:.2f}")
    logger.info(f"Validation SSIM: {val_metrics['SSIM']:.2f}")
    logger.info(f"Validation Pixel-to-Pixel Pearson CC: {val_metrics['Pearson']:.4f}")
    if args.binning > 1:
        logger.info(f"Validation MAE (binning): {val_metrics['MAE_binning']:.2f}")
        logger.info(f"Validation PSNR (binning): {val_metrics['PSNR_binning']:.2f}")
        logger.info(f"Validation SSIM (binning): {val_metrics['SSIM_binning']:.2f}")
        logger.info(f"Validation Pixel-to-Pixel Pearson CC (binning): {val_metrics['Pearson_binning']:.4f}")

    test_dir = log_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    test_metrics = calculate_metrics(
        model=G,
        dataset=test_dataset,
        loader=test_loader,
        device=device,
        log_dir=log_dir,
        output_dir=test_dir,
        args=args,
        stage="Test"
    )
    end_time = perf_counter()
    with open(log_dir / "test_metrics.yaml", "w") as file:
        yaml.dump(test_metrics, file)
    logger.info(f"Test time: {end_time - start_time:.2f} s, total {len(test_dataset)} samples")
    logger.info(f"Test MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Test PSNR: {test_metrics['PSNR']:.2f}")
    logger.info(f"Test SSIM: {test_metrics['SSIM']:.2f}")
    logger.info(f"Test Pixel-to-Pixel Pearson CC: {test_metrics['Pearson']:.4f}")
    if args.binning > 1:
        logger.info(f"Test MAE (binning): {test_metrics['MAE_binning']:.2f}")
        logger.info(f"Test PSNR (binning): {test_metrics['PSNR_binning']:.2f}")
        logger.info(f"Test SSIM (binning): {test_metrics['SSIM_binning']:.2f}")
        logger.info(f"Test Pixel-to-Pixel Pearson CC (binning): {test_metrics['Pearson_binning']:.4f}")





