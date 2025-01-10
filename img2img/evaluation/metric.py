import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from sunpy.map import Map

from torchmetrics.functional.regression import mean_absolute_error, pearson_corrcoef
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

import gc

from img2img.evaluation.sample import get_fake_target
from img2img.data.preprocess.sdo.preprocess_aia import NormalizeEditor
def denorm(x):
    x = NormalizeEditor(0, 14).inverse(x)
    return x

def get_mask_within_disk(smap, margin=0):
    Y, X = np.ogrid[:smap.data.shape[0], :smap.data.shape[1]]
    xc, yc = smap.wcs.world_to_pixel(smap.center)
    dist = np.sqrt((X-xc)**2 + (Y-yc)**2)
    mask = dist <= smap.meta['r_sun'] - margin  # Mask points inside the circle
    return mask

def calculate_metrics(model, cfg, dataset, loader, device, out_dir, csv_dir, args, stage):
    (out_dir/stage).mkdir(parents=True, exist_ok=True)

    metrics = {
        'Normalized_All': {'obstime': [], 'MAE': [], 'Pearson CC': [], 'PSNR': [], 'SSIM': []},
        'Normalized_WithinDisk': {'obstime': [], 'MAE': [], 'Pearson CC': [], 'PSNR': [], 'SSIM': []},
        'Normalized_OutsideDisk': {'obstime': [], 'MAE': [], 'Pearson CC': [], 'PSNR': [], 'SSIM': []},
        'Denormalized_All': {'obstime': [], 'MAE': [], 'Pearson CC': []},
        'Denormalized_WithinDisk': {'obstime': [], 'MAE': [], 'Pearson CC': []},
        'Denormalized_OutsideDisk': {'obstime': [], 'MAE': [], 'Pearson CC': []},
    }

    model.eval()
    with torch.no_grad():
        for i, (inputs, real_target, _, target_name) in enumerate(tqdm(loader, desc=stage)):
            assert inputs.size(0) == 1

            timestamp = target_name[0][:19]
            timestamp = timestamp[:10] + timestamp[10:].replace("-", ":")

            if args.save_meta:
                save_file_path = out_dir / stage / f"{target_name[0]}_fake.npz"
            else:
                save_file_path = out_dir / stage / f"{target_name[0]}_fake.npy"

            if not save_file_path.exists():
                inputs = inputs.to(device)
                fake_target = get_fake_target(model, cfg, args, inputs, device)
                fake_target = torch.clamp(fake_target, min=-1.0, max=1.0)

                if args.save_meta:
                    target_file = Path(dataset.target_dir) / (str(target_name[0]) + ".npz")
                    target_meta = np.load(target_file, allow_pickle=True)['metas']
                    np.savez(save_file_path, data=fake_target[0].cpu().numpy(), metas=target_meta)
                else:
                    # dataset.save_image(fake_target[0], out_dir / f"{target_name[0]}_fake.png")
                    # dataset.save_image(real_target[0], out_dir / f"{target_name[0]}_real.png"
                    np.save(save_file_path, fake_target[0].cpu().numpy())

            else:
                if args.save_meta:
                    fake_target = torch.from_numpy(np.load(save_file_path)['data']).unsqueeze(0)
                else:
                    fake_target = torch.from_numpy(np.load(save_file_path)).unsqueeze(0)
     
            real_target = real_target.cpu()
            fake_target = fake_target.cpu()
            real_target = torch.clamp(real_target, min=-1.0, max=1.0)
            fake_target = torch.clamp(fake_target, min=-1.0, max=1.0)

            if i == 0:
                fig = dataset.create_figure(inputs[0], real_target[0], fake_target[0])
                fig.savefig(csv_dir / f"{stage}_example.png")
                # print("Input        ", inputs.shape)
                # print("Target (Real)", real_target.shape)
                # print("Target (Fake)", fake_target.shape)

            mae = mean_absolute_error(fake_target, real_target)
            pcc = pearson_corrcoef(fake_target.flatten(), real_target.flatten())
            psnr = peak_signal_noise_ratio(fake_target, real_target, data_range=2.0)
            ssim = structural_similarity_index_measure(fake_target, real_target, data_range=2.0)
            metrics['Normalized_All']['obstime'].append(timestamp)
            metrics['Normalized_All']['MAE'].append(mae.item())
            metrics['Normalized_All']['Pearson CC'].append(pcc.item())
            metrics['Normalized_All']['PSNR'].append(psnr.item())
            metrics['Normalized_All']['SSIM'].append(ssim.item())

            if args.use_sunpy_map:
                target_file = Path(dataset.target_dir) / (str(target_name[0]) + ".npz")
                real_target_npz = np.load(target_file, allow_pickle=True)
                fake_target_npz = np.load(save_file_path, allow_pickle=True)
                real_target_map = Map(real_target_npz['data'][0], real_target_npz['metas'][0])
                fake_target_map = Map(fake_target_npz['data'][0], fake_target_npz['metas'][0])
            
                mask = torch.from_numpy(get_mask_within_disk(fake_target_map)).unsqueeze(0).unsqueeze(0)

                mae = mean_absolute_error(fake_target*mask, real_target*mask)
                pcc = pearson_corrcoef((fake_target*mask).flatten(), (real_target*mask).flatten())
                psnr = peak_signal_noise_ratio(fake_target*mask, real_target*mask, data_range=2.0)
                ssim = structural_similarity_index_measure(fake_target*mask, real_target*mask, data_range=2.0)
                metrics['Normalized_WithinDisk']['obstime'].append(timestamp)
                metrics['Normalized_WithinDisk']['MAE'].append(mae.item())
                metrics['Normalized_WithinDisk']['Pearson CC'].append(pcc.item())
                metrics['Normalized_WithinDisk']['PSNR'].append(psnr.item())
                metrics['Normalized_WithinDisk']['SSIM'].append(ssim.item())

                mae = mean_absolute_error(fake_target*~mask, real_target*~mask)
                pcc = pearson_corrcoef((fake_target*~mask).flatten(), (real_target*~mask).flatten())
                psnr = peak_signal_noise_ratio(fake_target*~mask, real_target*~mask, data_range=2.0)
                ssim = structural_similarity_index_measure(fake_target*~mask, real_target*~mask, data_range=2.0)
                metrics['Normalized_OutsideDisk']['obstime'].append(timestamp)
                metrics['Normalized_OutsideDisk']['MAE'].append(mae.item())
                metrics['Normalized_OutsideDisk']['Pearson CC'].append(pcc.item())
                metrics['Normalized_OutsideDisk']['PSNR'].append(psnr.item())
                metrics['Normalized_OutsideDisk']['SSIM'].append(ssim.item())
                
                # Denormalized
                real_target = torch.from_numpy(denorm(real_target_map).data).unsqueeze(0).unsqueeze(0)
                fake_target = torch.from_numpy(denorm(fake_target_map).data).unsqueeze(0).unsqueeze(0)

                mae = mean_absolute_error(fake_target, real_target)
                pcc = pearson_corrcoef(fake_target.flatten(), real_target.flatten())
                metrics['Denormalized_All']['obstime'].append(timestamp)
                metrics['Denormalized_All']['MAE'].append(mae.item())
                metrics['Denormalized_All']['Pearson CC'].append(pcc.item())

                mae = mean_absolute_error(fake_target*mask, real_target*mask)
                pcc = pearson_corrcoef((fake_target*mask).flatten(), (real_target*mask).flatten())
                metrics['Denormalized_WithinDisk']['obstime'].append(timestamp)
                metrics['Denormalized_WithinDisk']['MAE'].append(mae.item())
                metrics['Denormalized_WithinDisk']['Pearson CC'].append(pcc.item())

                mae = mean_absolute_error(fake_target*~mask, real_target*~mask)
                pcc = pearson_corrcoef((fake_target*~mask).flatten(), (real_target*~mask).flatten())
                metrics['Denormalized_OutsideDisk']['obstime'].append(timestamp)
                metrics['Denormalized_OutsideDisk']['MAE'].append(mae.item())
                metrics['Denormalized_OutsideDisk']['Pearson CC'].append(pcc.item())

            del inputs
            del real_target
            del fake_target
            del real_target_npz
            del fake_target_npz
            del real_target_map
            del fake_target_map

            gc.collect()
            torch.cuda.empty_cache()

            df_normalized_all = pd.DataFrame.from_dict(data=metrics['Normalized_All'])
            df_normalized_all.to_csv(csv_dir / f"{stage}_metrics_normalized_all.csv", index=False)

            if args.use_sunpy_map:
                df_normalized_within_disk = pd.DataFrame.from_dict(data=metrics['Normalized_WithinDisk'])
                df_normalized_outside_disk = pd.DataFrame.from_dict(data=metrics['Normalized_OutsideDisk'])
                df_denormalized_all = pd.DataFrame.from_dict(data=metrics['Denormalized_All'])
                df_denormalized_within_disk = pd.DataFrame.from_dict(data=metrics['Denormalized_WithinDisk'])
                df_denormalized_outside_disk = pd.DataFrame.from_dict(data=metrics['Denormalized_OutsideDisk'])

                
                df_normalized_within_disk.to_csv(csv_dir / f"{stage}_metrics_normalized_within_disk.csv", index=False)
                df_normalized_outside_disk.to_csv(csv_dir / f"{stage}_metrics_normalized_outside_disk.csv", index=False)
                df_denormalized_all.to_csv(csv_dir / f"{stage}_metrics_denormalized_all.csv", index=False)
                df_denormalized_within_disk.to_csv(csv_dir / f"{stage}_metrics_denormalized_within_disk.csv", index=False)
                df_denormalized_outside_disk.to_csv(csv_dir / f"{stage}_metrics_denormalized_outside_disk.csv", index=False)
        
        li = []
        ty = []
        li.append(df_normalized_all.drop('obstime', axis=1).mean())
        ty.append('Normalized_All')
        if args.use_sunpy_map:
            li.append(df_normalized_within_disk.drop('obstime', axis=1).mean())
            ty.append('Normalized_WithinDisk')
            li.append(df_normalized_outside_disk.drop('obstime', axis=1).mean())
            ty.append('Normalized_OutsideDisk')
            li.append(df_denormalized_all.drop('obstime', axis=1).mean())
            ty.append('Denormalized_All')
            li.append(df_denormalized_within_disk.drop('obstime', axis=1).mean())
            ty.append('Denormalized_WithinDisk')
            li.append(df_denormalized_outside_disk.drop('obstime', axis=1).mean())
            ty.append('Denormalized_OutsideDisk')
        
        df = pd.DataFrame(li, index=ty)
        df.index.name = 'Type'
        df.to_csv(csv_dir / f"{stage}_metrics.csv")
        return df