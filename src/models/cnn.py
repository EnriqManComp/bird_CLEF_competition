import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.block1 = self._set_cnn_blocks(1, 3, 32, 5, 0.3)

        self.pool = nn.MaxPool2d(2,2)

        self.block2 = self._set_cnn_blocks(2, 32, 64, 5, 0.5)

        self.block3 = self._set_cnn_blocks(3, 64, 128, 2, 0.3)

        self.block4 = nn.Sequential(
            nn.Linear(128*2*2, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.block1(x)))

        x = self.pool(F.relu(self.block2(x)))

        x = self.pool(F.relu(self.block3(x)))

        # Flat tensor
        x = x.view(-1, 128*2*2)

        x = self.block4(x)

        return x
    
    @staticmethod
    def _set_cnn_blocks(n_block:int, input_dim:int, n_hidden:int, output_dim:int, p_dropout:float):
        """ Set a Sequential Block in PyTorch """
        return nn.Sequential(
            nn.Conv2d(input_dim, n_hidden, output_dim),
            nn.BatchNorm2d(n_hidden),
            nn.Dropout(p_dropout)
        )
    
   