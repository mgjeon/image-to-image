python dpm/test.py --config /home/mgj/workspace/mgjeon/image-to-image/results/sdo/diffusion_ddpm/logs/version_0/hparams.yaml --model /home/mgj/workspace/mgjeon/image-to-image/results/sdo/diffusion_ddpm/logs/version_0/checkpoints/9_4819.pth --subsample 1 --timesteps 100

python dpm/test.py --config /home/mgj/workspace/mgjeon/image-to-image/results/sdo/diffusion_fast_ddpm/logs/version_0/hparams.yaml --model /home/mgj/workspace/mgjeon/image-to-image/results/sdo/diffusion_fast_ddpm/logs/version_0/checkpoints/9_4819.pth --subsample 1 --timesteps 10

python pix2pix/test.py --config /home/mgj/workspace/mgjeon/image-to-image/results/sdo/pix2pix_unet_patchgan/logs/version_0/hparams.yaml --model /home/mgj/workspace/mgjeon/image-to-image/results/sdo/pix2pix_unet_patchgan/logs/version_0/checkpoints/9_4819_G.pth --subsample 1