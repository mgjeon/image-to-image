python pix2pix/train.py --config pix2pix/configs/sdo/unet_patchgan_1024_small.yaml

python pix2pixHD/train.py --config pix2pixHD/configs/sdo/default_small.yaml

python pix2pixCC/train.py --config pix2pixCC/configs/sdo/default_small.yaml

python diffusion/train.py --config diffusion/configs/sdo/fast_ddpm_diffusers_noise.yaml

python diffusion/train.py --config diffusion/configs/sdo/fast_ddpm_diffusers_x0.yaml