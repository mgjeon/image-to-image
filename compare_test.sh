# python img2img/pix2pix/test.py --hparams results/sdo/pix2pix/unet_patchgan_1024/logs/version_0/hparams.yaml --config results/sdo/pix2pix/unet_patchgan_1024/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pix/unet_patchgan_1024/logs/version_0/checkpoints/49_48249_G.pth --log_root metrics/sdo/pix2pix/unet_patchgan_1024 --subsample -1 --binning -1 --save_meta

# python img2img/pix2pix/test.py --hparams results/sdo/pix2pix/unet_patchgan_1024/logs/version_0/hparams.yaml --config results/sdo/pix2pix/unet_patchgan_1024/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pix/unet_patchgan_1024/logs/version_0/checkpoints/49_48249_G_ema.pth --log_root metrics/sdo/pix2pix/unet_patchgan_1024_ema --subsample -1 --binning -1 --save_meta

# # -----------------------------------------------------------------------------------------

# python img2img/pix2pix/test.py --hparams results/sdo/pix2pix/unet_patchgan_1024_small/logs/version_0/hparams.yaml --config results/sdo/pix2pix/unet_patchgan_1024_small/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pix/unet_patchgan_1024_small/logs/version_0/checkpoints/49_48249_G.pth --log_root metrics/sdo/pix2pix/unet_patchgan_1024_small --subsample -1 --binning -1 --save_meta

# python img2img/pix2pix/test.py --hparams results/sdo/pix2pix/unet_patchgan_1024_small/logs/version_0/hparams.yaml --config results/sdo/pix2pix/unet_patchgan_1024_small/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pix/unet_patchgan_1024_small/logs/version_0/checkpoints/49_48249_G_ema.pth --log_root metrics/sdo/pix2pix/unet_patchgan_1024_small_ema --subsample -1 --binning -1 --save_meta

#=========================================================================================

python img2img/pix2pixHD/test.py --hparams results/sdo/pix2pixHD/default/logs/version_0/hparams.yaml --config results/sdo/pix2pixHD/default/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixHD/default/logs/version_0/checkpoints/49_48249_G.pth --log_root metrics/sdo/pix2pixHD/default --subsample -1 --binning -1 --save_meta

python img2img/pix2pixHD/test.py --hparams results/sdo/pix2pixHD/default/logs/version_0/hparams.yaml --config results/sdo/pix2pixHD/default/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixHD/default/logs/version_0/checkpoints/49_48249_G_ema.pth --log_root metrics/sdo/pix2pixHD/default_ema --subsample -1 --binning -1 --save_meta

#-----------------------------------------------------------------------------------------

python img2img/pix2pixHD/test.py --hparams results/sdo/pix2pixHD/default_small/logs/version_0/hparams.yaml --config results/sdo/pix2pixHD/default_small/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixHD/default_small/logs/version_0/checkpoints/49_48249_G.pth --log_root metrics/sdo/pix2pixHD/default_small --subsample -1 --binning -1 --save_meta

python img2img/pix2pixHD/test.py --hparams results/sdo/pix2pixHD/default_small/logs/version_0/hparams.yaml --config results/sdo/pix2pixHD/default_small/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixHD/default_small/logs/version_0/checkpoints/49_48249_G_ema.pth --log_root metrics/sdo/pix2pixHD/default_small_ema --subsample -1 --binning -1 --save_meta

#=========================================================================================

python img2img/pix2pixCC/test.py --hparams results/sdo/pix2pixCC/default/logs/version_0/hparams.yaml --config results/sdo/pix2pixCC/default/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixCC/default/logs/version_0/checkpoints/49_48249_G.pth --log_root metrics/sdo/pix2pixCC/default --subsample -1 --binning -1 --save_meta

python img2img/pix2pixCC/test.py --hparams results/sdo/pix2pixCC/default/logs/version_0/hparams.yaml --config results/sdo/pix2pixCC/default/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixCC/default/logs/version_0/checkpoints/49_48249_G_ema.pth --log_root metrics/sdo/pix2pixCC/default_ema --subsample -1 --binning -1 --save_meta

# -----------------------------------------------------------------------------------------

python img2img/pix2pixCC/test.py --hparams results/sdo/pix2pixCC/default_small/logs/version_0/hparams.yaml --config results/sdo/pix2pixCC/default_small/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixCC/default_small/logs/version_0/checkpoints/49_48249_G.pth --log_root metrics/sdo/pix2pixCC/default_small --subsample -1 --binning -1 --save_meta

python img2img/pix2pixCC/test.py --hparams results/sdo/pix2pixCC/default_small/logs/version_0/hparams.yaml --config results/sdo/pix2pixCC/default_small/logs/version_0/checkpoints/config.yaml --model results/sdo/pix2pixCC/default_small/logs/version_0/checkpoints/49_48249_G_ema.pth --log_root metrics/sdo/pix2pixCC/default_small_ema --subsample -1 --binning -1 --save_meta

#=========================================================================================

python img2img/diffusion/test.py --hparams results/sdo/diffusion/ddpm_noise/logs/version_0/hparams.yaml --config results/sdo/diffusion/ddpm_noise/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/ddpm_noise/logs/version_0/checkpoints/49_48249.pth --log_root metrics/sdo/diffusion/ddpm_noise --subsample -1 --binning -1 --save_meta

python img2img/diffusion/test.py --hparams results/sdo/diffusion/ddpm_noise/logs/version_0/hparams.yaml --config results/sdo/diffusion/ddpm_noise/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/ddpm_noise/logs/version_0/checkpoints/49_48249_ema.pth --log_root metrics/sdo/diffusion/ddpm_noise_ema --subsample -1 --binning -1 --save_meta

# -----------------------------------------------------------------------------------------

python img2img/diffusion/test.py --hparams results/sdo/diffusion/ddpm_x0/logs/version_0/hparams.yaml --config results/sdo/diffusion/ddpm_x0/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/ddpm_x0/logs/version_0/checkpoints/49_48249.pth --log_root metrics/sdo/diffusion/ddpm_x0 --subsample -1 --binning -1 --save_meta

python img2img/diffusion/test.py --hparams results/sdo/diffusion/ddpm_x0/logs/version_0/hparams.yaml --config results/sdo/diffusion/ddpm_x0/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/ddpm_x0/logs/version_0/checkpoints/49_48249_ema.pth --log_root metrics/sdo/diffusion/ddpm_x0_ema --subsample -1 --binning -1 --save_meta

# =========================================================================================

python img2img/diffusion/test.py --hparams results/sdo/diffusion/fast_ddpm_noise/logs/version_0/hparams.yaml --config results/sdo/diffusion/fast_ddpm_noise/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/fast_ddpm_noise/logs/version_0/checkpoints/49_48249.pth --log_root metrics/sdo/diffusion/fast_ddpm_noise --subsample -1 --binning -1 --save_meta

python img2img/diffusion/test.py --hparams results/sdo/diffusion/fast_ddpm_noise/logs/version_0/hparams.yaml --config results/sdo/diffusion/fast_ddpm_noise/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/fast_ddpm_noise/logs/version_0/checkpoints/49_48249_ema.pth --log_root metrics/sdo/diffusion/fast_ddpm_noise_ema --subsample -1 --binning -1 --save_meta

# -----------------------------------------------------------------------------------------

python img2img/diffusion/test.py --hparams results/sdo/diffusion/fast_ddpm_x0/logs/version_0/hparams.yaml --config results/sdo/diffusion/fast_ddpm_x0/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/fast_ddpm_x0/logs/version_0/checkpoints/49_48249.pth --log_root metrics/sdo/diffusion/fast_ddpm_x0 --subsample -1 --binning -1 --save_meta

python img2img/diffusion/test.py --hparams results/sdo/diffusion/fast_ddpm_x0/logs/version_0/hparams.yaml --config results/sdo/diffusion/fast_ddpm_x0/logs/version_0/checkpoints/config.yaml --model results/sdo/diffusion/fast_ddpm_x0/logs/version_0/checkpoints/49_48249_ema.pth --log_root metrics/sdo/diffusion/fast_ddpm_x0_ema --subsample -1 --binning -1 --save_meta