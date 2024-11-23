import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--pth", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    net_G_weights = {k.replace("net_G.",''): v for k, v in ckpt["state_dict"].items() if k.startswith("net_G.")}
    torch.save(net_G_weights, args.pth)
