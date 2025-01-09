import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

from torchmetrics import MeanAbsoluteError
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import gc

from img2img.evaluation.gan import get_fake_target

def calculate_metrics(model, cfg, dataset, loader, device, out_dir, csv_dir, args, stage):
    model.eval()
    with torch.no_grad():
        mae = MeanAbsoluteError().to(device)                                # 0.0 is best
        psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)              # +inf is best
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)  # 1.0 is best
        pearson = PearsonCorrCoef().to(device)                              # 1.0 is best
        
        maes = []
        psnrs = []
        ssims = []
        pearsons = []

        for i, (inputs, real_target, _, target_name) in enumerate(tqdm(loader, desc=stage)):
            inputs = inputs.to(device)
            real_target = real_target.to(device)
            fake_target = get_fake_target(model, cfg, inputs, device)

            real_target = torch.clamp(real_target, min=-1.0, max=1.0)
            fake_target = torch.clamp(fake_target, min=-1.0, max=1.0)

            if i == 0:
                fig = dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                fig.savefig(csv_dir / f"{stage}_example.png")
                # print("Input        ", inputs.shape)
                # print("Target (Real)", real_target.shape)
                # print("Target (Fake)", fake_target.shape)

            bs = inputs.size(0)

            for i in range(bs):
                if args.save_meta:
                    target_file = Path(dataset.target_dir) / target_name[i]
                    target_file = str(target_file) + ".npz"
                    target_meta = np.load(target_file, allow_pickle=True)['metas']
                    np.savez(out_dir / f"{target_name[i]}_fake.npz", data=fake_target[i].cpu().numpy(), metas=target_meta)
                else:
                    dataset.save_image(fake_target[0], out_dir / f"{target_name[0]}_fake.png")
                    dataset.save_image(real_target[0], out_dir / f"{target_name[0]}_real.png")
                
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
            
            del inputs
            del real_target
            del fake_target

            gc.collect()
            torch.cuda.empty_cache()

        mae = sum(maes) / len(maes)
        psnr = sum(psnrs) / len(psnrs)
        ssim = sum(ssims) / len(ssims)
        pearson = sum(pearsons) / len(pearsons)

        res = {
            "MAE": [mae],
            "PSNR": [psnr],
            "SSIM": [ssim],
            "Pearson CC": [pearson]
        }

        df = pd.DataFrame.from_dict(data=res)
        df.to_csv(csv_dir / f"{stage}_metrics.csv", index=False)
        return df
