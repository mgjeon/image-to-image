python -m img2img.eval --ema --config configs/default/gan/pix2pix.yaml --output_dir results/sdo/gan/pix2pix --dataset_root data/aia_dataset --device 0 --ckpt last
python -m img2img.eval --ema --config configs/default/gan/pix2pix.yaml --output_dir results/sdo/gan/pix2pix --dataset_root data/aia_dataset --device 0 --ckpt best
python -m img2img.eval --ema --config configs/default/gan/pix2pix.yaml --output_dir results/sdo/gan/pix2pix --dataset_root data/aia_dataset --device 0 --ckpt epoch --epoch 49

python -m img2img.eval --ema --config configs/default/gan/pix2pixhd.yaml --output_dir results/sdo/gan/pix2pixhd --dataset_root data/aia_dataset --device 0 --ckpt last
python -m img2img.eval --ema --config configs/default/gan/pix2pixhd.yaml --output_dir results/sdo/gan/pix2pixhd --dataset_root data/aia_dataset --device 0 --ckpt best
python -m img2img.eval --ema --config configs/default/gan/pix2pixhd.yaml --output_dir results/sdo/gan/pix2pixhd --dataset_root data/aia_dataset --device 0 --ckpt epoch --epoch 49

python -m img2img.eval --ema --config configs/default/gan/pix2pixcc.yaml --output_dir results/sdo/gan/pix2pixcc --dataset_root data/aia_dataset --device 0 --ckpt last
python -m img2img.eval --ema --config configs/default/gan/pix2pixcc.yaml --output_dir results/sdo/gan/pix2pixcc --dataset_root data/aia_dataset --device 0 --ckpt best
python -m img2img.eval --ema --config configs/default/gan/pix2pixcc.yaml --output_dir results/sdo/gan/pix2pixcc --dataset_root data/aia_dataset --device 0 --ckpt epoch --epoch 49