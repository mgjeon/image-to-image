# python train.py --config configs/sdo/unet_patchgan.yaml

# python train_lightning.py --config configs/sdo/unet_patchgan.yaml

# python train.py --config configs/sdo/unet_patchgan.yaml --resume_from /userhome/jeon_mg/workspace/image-to-image/pix2pix/results/sdo/unet_patchgan/logs/version_0/checkpoints/1_239_checkpoint.pth --num_epochs 4

# python train_lightning.py --config configs/sdo/unet_patchgan.yaml --resume_from /userhome/jeon_mg/workspace/image-to-image/pix2pix/results/sdo/unet_patchgan/logs/version_1/checkpoints/epoch=1.ckpt --num_epochs 4

# python test.py --config /userhome/jeon_mg/workspace/image-to-image/pix2pix/results/sdo/unet_patchgan/logs/version_2/hparams.yaml --model /userhome/jeon_mg/workspace/image-to-image/pix2pix/results/sdo/unet_patchgan/logs/version_2/checkpoints/3_479_G.pth --device 3

python test.py --config /userhome/jeon_mg/workspace/image-to-image/pix2pix/results/sdo/unet_patchgan/logs/version_3/hparams.yaml --model /userhome/jeon_mg/workspace/image-to-image/pix2pix/results/sdo/unet_patchgan/logs/version_3/checkpoints/3_239_G.pth --device 2