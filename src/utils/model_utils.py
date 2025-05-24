import torch

def get_accuracy(prediction: str, label: str) -> float:
    
    matches = [i == j for i,j in zip(prediction, label)]
    
    accuracy = matches.count(True) / len(matches)
    return accuracy