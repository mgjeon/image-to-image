python -m img2img.train --config configs/default/gan/pix2pix.yaml
python -m img2img.train --config configs/default/gan/pix2pixhd.yaml
python -m img2img.train --config configs/default/gan/pix2pixcc.yaml
python -m img2img.train --config configs/default/diffusion/ddpm_x0.yaml
python -m img2img.train --config configs/default/diffusion/fast_ddpm_x0.yaml
python -m img2img.train --config configs/default/diffusion/ddpm_noise.yaml
python -m img2img.train --config configs/default/diffusion/fast_ddpm_noise.yaml

python -m img2img.test --config configs/default/gan/pix2pix.yaml
python -m img2img.test --config configs/default/gan/pix2pixhd.yaml
python -m img2img.test --config configs/default/gan/pix2pixcc.yaml
python -m img2img.test --config configs/default/diffusion/ddpm_x0.yaml
python -m img2img.test --config configs/default/diffusion/fast_ddpm_x0.yaml
python -m img2img.test --config configs/default/diffusion/ddpm_noise.yaml
python -m img2img.test --config configs/default/diffusion/fast_ddpm_noise.yaml