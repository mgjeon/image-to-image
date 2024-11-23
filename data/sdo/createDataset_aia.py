"""
@author: Mingyu Jeon (mgjeon@khu.ac.kr)
"""
import yaml
import argparse

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
from sunpy.map import Map
from tqdm import tqdm

from preprocess_aia import get_intersecting_files

def create_input_target(
    original_path,
    dataset_root,
    kind='train',
    input_wavelengths=[171, 304],
    target_wavelengths=[193],
    months=[1],
    years=[2020, 2021, 2022, 2023],
):
    root = Path(dataset_root) / kind
    input_root = root / 'input'
    target_root = root / 'target'
    input_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    input_files = get_intersecting_files(
        path=original_path,
        dirs=input_wavelengths,
        months=months,
        years=years
    )

    target_files = get_intersecting_files(
        path=original_path,
        dirs=target_wavelengths,
        months=months,
        years=years
    )

    input_wl = '_'.join(map(str, input_wavelengths))
    target_wl = '_'.join(map(str, target_wavelengths))

    for con in tqdm(
        zip(*input_files), 
        total=len(input_files[0]),
        desc=f'Creating {kind} input data',
    ):
        save_path = input_root / (Path(con[0]).stem + '_' + input_wl +'.npz')
        s_maps = [Map(file) for file in con]
        datas = [s_map.data for s_map in s_maps]
        metas = [s_map.meta for s_map in s_maps]
        # [C, H, W]
        stacked = np.stack(datas, axis=0)
        np.savez(save_path, data=stacked, metas=metas)

    for con in tqdm(
        zip(*target_files), 
        total=len(target_files[0]),
        desc=f'Creating {kind} target data',
    ):
        save_path = target_root / (Path(con[0]).stem + '_' + target_wl + '.npz')
        s_maps = [Map(file) for file in con]
        datas = [s_map.data for s_map in s_maps]
        metas = [s_map.meta for s_map in s_maps]
        # [C, H, W]
        stacked = np.stack(datas, axis=0)
        np.savez(save_path, data=stacked, metas=metas)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/createDataset_aia.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    original_root = config['original_root']
    dataset_root = config['dataset_root']

    create_input_target(
        original_root,
        dataset_root,
        kind='train',
        input_wavelengths=config['train']['input_wavelengths'],
        target_wavelengths=config['train']['target_wavelengths'],
        months=config['train']['months'],
        years=config['train']['years'],
    )

    create_input_target(
        original_root,
        dataset_root,
        kind='val',
        input_wavelengths=config['val']['input_wavelengths'],
        target_wavelengths=config['val']['target_wavelengths'],
        months=config['val']['months'],
        years=config['val']['years'],
    )

    create_input_target(
        original_root,
        dataset_root,
        kind='test',
        input_wavelengths=config['test']['input_wavelengths'],
        target_wavelengths=config['test']['target_wavelengths'],
        months=config['test']['months'],
        years=config['test']['years'],
    )