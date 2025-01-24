python -m img2img.eval --config configs/default/diffusion/ddpm_x0.yaml --timesteps 10 --output_dir results/sdo/diffusion/ddpm/x0 --dataset_root data/aia_dataset --device 0 --ckpt last
python -m img2img.eval --config configs/default/diffusion/ddpm_x0.yaml --timesteps 10 --output_dir results/sdo/diffusion/ddpm/x0 --dataset_root data/aia_dataset --device 0 --ckpt best
python -m img2img.eval --config configs/default/diffusion/ddpm_x0.yaml --timesteps 10 --output_dir results/sdo/diffusion/ddpm/x0 --dataset_root data/aia_dataset --device 0 --ckpt epoch --epoch 49

python -m img2img.eval --config configs/default/diffusion/fast_ddpm_x0.yaml --timesteps 10 --output_dir results/sdo/diffusion/fast_ddpm/x0 --dataset_root data/aia_dataset --device 0 --ckpt last
python -m img2img.eval --config configs/default/diffusion/fast_ddpm_x0.yaml --timesteps 10 --output_dir results/sdo/diffusion/fast_ddpm/x0 --dataset_root data/aia_dataset --device 0 --ckpt best
python -m img2img.eval --config configs/default/diffusion/fast_ddpm_x0.yaml --timesteps 10 --output_dir results/sdo/diffusion/fast_ddpm/x0 --dataset_root data/aia_dataset --device 0 --ckpt epoch --epoch 49

python -m img2img.eval --config configs/default/diffusion/ddpm_noise.yaml --timesteps 10 --output_dir results/sdo/diffusion/ddpm/noise --dataset_root data/aia_dataset --device 0 --ckpt last
python -m img2img.eval --config configs/default/diffusion/ddpm_noise.yaml --timesteps 10 --output_dir results/sdo/diffusion/ddpm/noise --dataset_root data/aia_dataset --device 0 --ckpt best
python -m img2img.eval --config configs/default/diffusion/ddpm_noise.yaml --timesteps 10 --output_dir results/sdo/diffusion/ddpm/noise --dataset_root data/aia_dataset --device 0 --ckpt epoch --epoch 49 

python -m img2img.eval --config configs/default/diffusion/fast_ddpm_noise.yaml --timesteps 10 --output_dir results/sdo/diffusion/fast_ddpm/noise --dataset_root data/aia_dataset --device 0 --ckpt last
python -m img2img.eval --config configs/default/diffusion/fast_ddpm_noise.yaml --timesteps 10 --output_dir results/sdo/diffusion/fast_ddpm/noise --dataset_root data/aia_dataset --device 0 --ckpt best
python -m img2img.eval --config configs/default/diffusion/fast_ddpm_noise.yaml --timesteps 10 --output_dir results/sdo/diffusion/fast_ddpm/noise --dataset_root data/aia_dataset --device 0 --ckpt epoch --epoch 49