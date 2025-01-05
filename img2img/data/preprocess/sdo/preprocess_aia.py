"""
@author: Mingyu Jeon (mgjeon@khu.ac.kr)

Adapted from the ITI code by Robert Jarolim
Reference:
1) https://github.com/RobertJaro/InstrumentToInstrument
"""

import os
import glob
import yaml
import inspect
import logging
import argparse
from pathlib import Path
from dateutil.parser import parse
from abc import ABC,  abstractmethod
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sunpy.map import Map
import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
# from astropy.visualization import ImageNormalize, AsinhStretch
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
from tqdm import tqdm


def get_intersecting_files(
    path, 
    dirs, 
    months=None, 
    years=None, 
    n_samples=None, 
    ext=None, 
    basenames=None, 
    **kwargs
):
    """
    Get intersecting files from multiple directories

    Args:
        path (str): Path to directories
        dirs (list): List of directories
        months (list): List of months
        years (list): List of years
        n_samples (int): Number of samples
        ext (str): File extension
        basenames (list): List of basenames
        **kwargs: Additional arguments

    Returns:
        list: List of intersecting files
    """
    pattern = '*' if ext is None else '*' + ext
    if basenames is None:
        basenames = [
            [os.path.basename(path) for path in glob.glob(os.path.join(path, str(d), '**', pattern), recursive=True)]
            for d in dirs]
        basenames = list(set(basenames[0]).intersection(*basenames))
    if months:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('.')[0]).month in months]
    if years:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('.')[0]).year in years]
    basenames = sorted(list(basenames))
    if n_samples:
        basenames = basenames[::len(basenames) // n_samples]
    return [[os.path.join(path, str(dir), b) for b in basenames] for dir in dirs]


def get_local_correction_table():
    """
    Get local correction table for AIA data

    Args:
        overwrite (bool): overwrite the existing correction

    Returns:
        correction_table (pd.DataFrame): correction table
    """
    path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
    if os.path.exists(path):
        return get_correction_table(correction_table=path)
    os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
    correction_table = get_correction_table()
    ascii.write(correction_table, path)
    return correction_table


# sdo_norms = {
#     171: ImageNormalize(vmin=0, vmax=8600, 
#                         stretch=AsinhStretch(0.005), 
#                         clip=False),
#     193: ImageNormalize(vmin=0, vmax=9800, 
#                         stretch=AsinhStretch(0.005),
#                         clip=False),
#     304: ImageNormalize(vmin=0, vmax=8800, 
#                         stretch=AsinhStretch(0.001), 
#                         clip=False),
# }

class Normalizer:
    
    def __init__(self, LoLim, UpLim):
        if LoLim == 0:
            l_lim = 0
        else:
            l_lim = 2**LoLim-1
        u_lim = 2**UpLim-1

        self.l_lim = l_lim
        self.u_lim = u_lim
        self.LoLim = LoLim
        self.UpLim = UpLim
    
    def normalize(self, data):
        l_lim = self.l_lim
        u_lim = self.u_lim
        LoLim = self.LoLim
        UpLim = self.UpLim
        data = np.clip(data, l_lim, u_lim)                       # data     -> [2**LoLim-1, 2**UpLim-1]
        data = np.log2(data + 1)                                 # data + 1 -> [LoLim, UpLim]
        data = (data -((UpLim + LoLim)/2))/((UpLim - LoLim)/2)   # data + 1 -> [-1, 1]
        return data
    
    def inverse(self, data):
        LoLim = self.LoLim
        UpLim = self.UpLim
        data = data * ((UpLim - LoLim)/2) + ((UpLim + LoLim)/2)  # data -> [LoLim, UpLim]
        data = 2**data - 1                                       # data -> [2**LoLim-1, 2**UpLim-1]
        return data

#===============================================================================
class Editor(ABC):
    """
    Editor class for data processing
    """

    @abstractmethod
    def call(self, data, **kwargs):
        raise NotImplementedError()


class LoadMapEditor(Editor):
    """
    Load SunPy Map editor

    Args:
        data (str): FITS file path

    Returns:
        s_map (sunpy.map.Map): SunPy Map object
        path (str): file path
    """
    def call(self, data, **kwargs):
        s_map = Map(data)
        s_map.meta['timesys'] = 'tai'  # fix leap seconds
        return s_map
        

class NormalizeRadiusEditor(Editor):
    """
    Normalize radius editor cropping and padding the image to a fixed resolution to 1.28 solar radii (= 400 pix for 1024 resolution)

    Args:
        resolution (int): resolution
        radius_padding_factor (float): specify the solar radius padding factor
        crop (bool): crop
        rotate_north_up (bool): rotate north up
        fix_irradiance_with_distance (bool): fix irradiance with distance
        s_map (sunpy.map.Map): SunPy Map object

    Returns:
        s_map (sunpy.map.Map): SunPy Map object
    """
    def __init__(
        self,
        resolution,
        radius_padding_factor=0.28,
        rotate_north_up=True,
        crop=True,
        fix_irradiance_with_distance=False,
        logger=None,
        **kwargs
    ):
        self.resolution = resolution
        self.padding_factor = radius_padding_factor
        self.rotate_north_up = rotate_north_up
        self.crop = crop
        self.fix_irradiance_with_distance = fix_irradiance_with_distance
        self.logger = logger if logger else logging.getLogger('AIAPreprocessor')
        super().__init__(**kwargs)
    
    def call(self, s_map, **kwargs):

        original_map = s_map

        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # Get the solar radius in pixels
        self.logger.info(f"Solar radius : {r_obs_pix.value:.2f} pix => {(self.resolution / 2) / (1 + self.padding_factor):.2f} pix = 1 solar radii")
        self.logger.info(f"Resolution/2 : {s_map.data.shape[0]/2:.2f} pix => {self.resolution/2:.2f} pix = {1 + self.padding_factor:.2f} solar radii")
        r_obs_pix = (1 + self.padding_factor) * r_obs_pix  # Get the size in pixels of the padded radius
        scale_factor = self.resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)

        # Rotate & Scale
        if self.rotate_north_up:
            s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=4)
        else:
            s_map = s_map.rotate(angle=0*u.deg, recenter=False, scale=scale_factor, missing=0, order=4)
        
        # Crop
        if self.crop:
            arcs_frame = (self.resolution / 2) * s_map.scale[0].value
            s_map = s_map.submap(
                bottom_left=SkyCoord(-arcs_frame * u.arcsec, -arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
            pad_x = s_map.data.shape[0] - self.resolution
            pad_y = s_map.data.shape[1] - self.resolution
            s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                                 top_right=[pad_x // 2 + self.resolution - 1, pad_y // 2 + self.resolution - 1] * u.pix)
            
        # pad with zeros if the map is too small
        if s_map.data.shape[0] < self.resolution or s_map.data.shape[1] < self.resolution:
            data = s_map.data
            new_data = np.zeros((self.resolution, self.resolution))
            padding_x = (self.resolution - data.shape[0]) // 2
            padding_y = (self.resolution - data.shape[1]) // 2
            new_data[padding_x:padding_x + data.shape[0], padding_y:padding_y + data.shape[1]] = data
            s_map = Map(new_data, s_map.meta)
        
        # Update metadata with new resolution
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']

        # Virtually move the instrument such that the sun occupies the expected
        # size in the current optics
        # (s_map)*(1AU)^2 = (original_map)*(dsun)^2
        if self.fix_irradiance_with_distance:
            # preserve total intensity at 1 AU
            s_map.data[:] = s_map.data * (original_map.data.sum() / s_map.data.sum()) * (original_map.dsun.to_value(u.AU) / 1)**2
            # set radius to 1 AU
            s_map.meta['dsun_obs'] = (1 * u.AU).to_value(u.m)
        return s_map
    

class AIACorrectDegradationEditor(Editor):
    """
    AIA data preparation editor for instrument degradation correction

    Args:
        calibration (str): calibration
        s_map (sunpy.map.Map): SunPy Map object
        correction_table (pd.DataFrame): correction table

    Returns:
        s_map (sunpy.map.Map): SunPy Map object
    """
    def __init__(self, calibration='aiapy'):
        super().__init__()
        assert calibration in ['aiapy', 'none',
                               None], "Calibration must be one of: ['aiapy', 'none', None]"
        self.calibration = calibration
        self.table = get_local_correction_table()

    def call(self, s_map, **kwargs):
        if self.calibration == 'aiapy':
            s_map = correct_degradation(s_map, correction_table=self.table)
        return s_map
    

class AIADivideExposureEditor(Editor):
    """
    AIA data preparation editor for dividing by exposure time

    Args:
        s_map (sunpy.map.Map): SunPy Map object

    Returns:
        s_map (sunpy.map.Map): SunPy Map object
    """
    def __init__(self):
        super().__init__()

    def call(self, s_map, **kwargs):
        data = np.nan_to_num(s_map.data)
        data = data / s_map.meta["exptime"]
        s_map.meta['bunit'] = 'DN/s'
        s_map = Map(data.astype(np.float32), s_map.meta)
        return s_map
    
    def inverse(self, s_map, **kwargs):
        data = np.nan_to_num(s_map.data)
        data = data * s_map.meta["exptime"]
        s_map.meta['bunit'] = 'DN'
        s_map = Map(data.astype(np.float32), s_map.meta)
        return s_map
    

# class NormalizeEditor(Editor):
#     """
#     Normalize data editor in range [-1, 1]

#     Args:
#         norm (astropy.visualization.ImageNormalize): normalization function in range [0, 1]
#         s_map (sunpy.map.Map): SunPy Map object

#     Returns:
#         (sunpy.map.Map): SunPy Map object
#     """
#     def __init__(self, norm):
#         self.norm = norm

#     def call(self, s_map, **kwargs):
#         data = s_map.data
#         data = self.norm(data).data * 2 - 1
#         s_map = Map(data.astype(np.float32), s_map.meta)
#         return s_map
    
#     def inverse(self, s_map, **kwargs):
#         data = s_map.data
#         data = (data + 1) / 2
#         data = self.norm.inverse(data)
#         s_map = Map(data.astype(np.float32), s_map.meta)
#         return s_map

class NormalizeEditor(Editor):
    """
    Normalize data editor in range [-1, 1]

    Args:
        LoLim (int): Lower limit 
        UpLim (int): Upper limit
        s_map (sunpy.map.Map): SunPy Map object

    Returns:
        (sunpy.map.Map): SunPy Map object
    """
    def __init__(self, LoLim, UpLim):
        self.norm = Normalizer(LoLim, UpLim)

    def call(self, s_map, **kwargs):
        data = s_map.data
        data = self.norm.normalize(data)
        s_map = Map(data.astype(np.float32), s_map.meta)
        return s_map
    
    def inverse(self, s_map, **kwargs):
        data = s_map.data
        data = self.norm.inverse(data)
        s_map = Map(data.astype(np.float32), s_map.meta)
        return s_map
#===============================================================================


class SDOAIAPreprocessor:
    def __init__(
        self,
        original_path,
        preprocessed_path,
        wavelengths=[171, 193, 304],
        months=None,
        years=None,
        ext='.fits',
        resolution=1024,
        radius_padding_factor=0.28,
        rotate_north_up=True,
        crop=True,
        calibration='aiapy',
        LoLim=0,  
        UpLim=14,  # data range [2**LoLim-1, 2**UpLim-1] DN/s
    ):  
        self.paths = get_intersecting_files(
            path=original_path, 
            dirs=wavelengths, 
            months=months, 
            years=years, 
            ext=ext, 
        )
        assert len(self.paths) == len(wavelengths)
        self.preprocessed_path = Path(preprocessed_path)
        self.preprocessed_path.mkdir(parents=True, exist_ok=True)
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.radius_padding_factor = radius_padding_factor
        self.rotate_north_up = rotate_north_up
        self.crop = crop
        self.calibration = calibration
        self.setup_logging()
        self.editors = [
            LoadMapEditor(),
            NormalizeRadiusEditor(
                resolution=resolution,
                radius_padding_factor=radius_padding_factor,
                rotate_north_up=rotate_north_up,
                crop=crop,
                logger=self.logger
            ),
            AIACorrectDegradationEditor(calibration=calibration),
            AIADivideExposureEditor(),
        ]
        self.editors = {wl:self.editors + [NormalizeEditor(LoLim, UpLim)] for wl in wavelengths}

        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        with open(self.preprocessed_path / 'config.yaml', 'w') as file:
            yaml.dump(local_vars['config'], file)
    
    def setup_logging(self):
        log_file = self.preprocessed_path / 'log.txt'
        logger = logging.getLogger('AIAPreprocessor')
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        self.logger = logger


    def run(
        self,
        **kwargs
    ):
        n_wavelengths = len(self.wavelengths)
        tqdm_loader = tqdm(range(n_wavelengths), desc='Preprocessing')
        for i in tqdm_loader:
            files = self.paths[i]
            assert str(Path(files[0]).parent.name) == str(self.wavelengths[i])
            wavelength = self.wavelengths[i]
            self.logger.info(f"Preprocessing {wavelength} AIA data")
            save_path = self.preprocessed_path / str(wavelength)
            save_path.mkdir(parents=True, exist_ok=True)
            tqdm_file = tqdm(files, leave=True, desc=f'Wavelength {wavelength}')
            for file in tqdm_file:
                try:
                    save_file = save_path / Path(file).name
                    if save_file.exists():
                        try:
                            Map(save_file)
                            tqdm_file.set_postfix_str(f"File {save_file} already exists. Skipping...")
                            continue
                        except Exception as e:
                            tqdm_file.set_postfix_str(f"File {save_file} is corrupted. Overwriting...")
                    self.logger.info(f"Preprocessing {file}")
                    data = self.preprocess(file, wavelength, **kwargs)
                    data.save(save_file, filetype='fits', overwrite=True)
                    tqdm_file.set_postfix_str(f"Saved to {save_file}")
                except Exception as e:
                    self.logger.error(f"Error preprocessing {file}: {e}")
                    continue
        
    
    def preprocess(
        self,
        data,
        wavelength,
        **kwargs
    ):
        for editor in self.editors[wavelength]:
            data = editor.call(data, **kwargs)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/preprocess_aia.yaml")

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    preprocessor = SDOAIAPreprocessor(**config)
    preprocessor.run()
