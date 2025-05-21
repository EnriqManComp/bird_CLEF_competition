import os
import random
import numpy as np
import librosa
import torch

root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

def create_dir(dir: str) -> None:
    if os.path.isdir(dir):
        print(dir, 'already exists ...')
    else:
        print('Making a new directory: ', dir)
        os.makedirs(dir)

def get_accuracy(prediction: str, label: str) -> float:
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(prediction, label)]
    accuracy = matches.count(True) / len(matches)
    return accuracy