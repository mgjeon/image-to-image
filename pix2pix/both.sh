# python train.py --config configs/sdo/simple.yaml
# python train.py --config configs/sdo/resnet_patchgan.yaml
# python train.py --config configs/sdo/unet_patchgan.yaml
# python train_pytorch.py --config configs/sdo/simple_pytorch.yaml

# python test.py --config /home/mgj/workspace/mgjeon/image-to-image/pix2pix/results/sdo/default_lightning/logs/version_0/hparams.yaml --checkpoint /home/mgj/workspace/mgjeon/image-to-image/pix2pix/results/sdo/default_lightning/logs/version_0/checkpoints/49_12050_G.pth

python train.py --config configs/sdo/simple.yaml
python train_fabric.py --config configs/sdo/simple.yaml
python train_lightning.py --config configs/sdo/simple.yaml