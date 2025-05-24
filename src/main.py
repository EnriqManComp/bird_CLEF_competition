import torch
import os
from models.train import train
from models.test import test
import argparse

def main(mode:str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set training parameters
    config_args = {
        "epochs": 20,
        "batch_size": 16,
        "lr": 0.001,
        "workers": 0,
        "device": device
    }

    root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    if mode == 'train':
        train(root_dir=root_dir, **config_args)
    else:
        test(root_dir=root_dir, **config_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose between training and testing stages")

    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                               help="Choose between train or test options")
    args = parser.parse_args()
    main(args.mode)



