# python pix2pix/train.py --config pix2pix/configs/sdo/unet_patchgan_1024_small.yaml

# python pix2pixHD/train.py --config pix2pixHD/configs/sdo/default_small.yaml

# python pix2pixCC/train.py --config pix2pixCC/configs/sdo/default_small.yaml


# python pix2pix/train.py --config pix2pix/configs/sdo/unet_patchgan_1024.yaml

# python pix2pixHD/train.py --config pix2pixHD/configs/sdo/default.yaml

# python pix2pixCC/train.py --config pix2pixCC/configs/sdo/default.yaml


# python diffusion/train.py --config diffusion/configs/sdo/fast_ddpm_diffusers_noise.yaml

# python diffusion/train.py --config diffusion/configs/sdo/fast_ddpm_diffusers_x0.yaml

# python diffusion/train.py --config diffusion/configs/sdo/ddpm_diffusers_noise.yaml

# python diffusion/train.py --config diffusion/configs/sdo/ddpm_diffusers_x0.yaml


python img2img/diffusion/train.py --config img2img/diffusion/configs/sdo/ddpm_uvit_noise.yaml

python img2img/diffusion/train.py --config img2img/diffusion/configs/sdo/ddpm_uvit_x0.yaml

python img2img/diffusion/train.py --config img2img/diffusion/configs/sdo/fast_ddpm_uvit_noise.yaml

python img2img/diffusion/train.py --config img2img/diffusion/configs/sdo/fast_ddpm_uvit_x0.yaml