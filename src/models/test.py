import os

import torchvision
from torchvision.transforms import transforms
import torch
import torch.nn as nn
from models.cnn import CNN

from datetime import date
import logging

from utils.utils import create_dir
from utils.model_utils import get_accuracy

def test(root_dir, **kwargs):

    # Create training session
    today = date.today()
    today = str(today.strftime('%m-%d-%Y'))
    dir_ = str(root_dir + '/bird_CLEF_competition/models/human_voice/CNN/train-' + today)
    create_dir(dir_)

    log_file_name = 'CNN-test' + today + '.log'
    logging.basicConfig(filename=os.path.join(dir_, log_file_name),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        level=logging.INFO
                        )

    # In this project we only use resize and convert to tensor
    # because change the spectogram change the meaning of the features

    transform = {
        'test': transforms.Compose([transforms.Resize([32,32]),
                                     transforms.ToTensor()])
    }

    # Loading testing data
    test_data = torchvision.datasets.ImageFolder(root=root_dir + '/bird_CLEF_competition/data/test_images/',
                                                  transform=transform['test'])

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=kwargs["workers"]
    )

    model = CNN()
    model = model.to(kwargs["device"])
    model_file_name = 'CNN-' + today + '.pt'
    model.load_state_dict(torch.load(os.path.join(dir_, model_file_name)))

    loss_fx = nn.BCELoss()

    # Testing Loop
    
    model.eval()
    loss = 0.0
    accuracy = 0.0
    performance = compute_performace(accuracy, loss)
        

    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(kwargs["device"]), label.to(kwargs["device"]).float().view(-1,1)

        prediction = model(img)
        b_prediction = nn.functional.sigmoid(prediction)
        
        loss = loss_fx(b_prediction, label)
        b_prediction = (b_prediction > 0.5).int()

        accuracy = get_accuracy(b_prediction.squeeze(-1), label.squeeze(-1).int())

        loss += loss.item()

    performance = compute_performace(accuracy, loss)
    # Testing status

    print('Loss: %.3f | Accuracy: %.3f | Performance: %.3f' % (        
        loss,
        accuracy,
        performance
    ))

    logging.info('Loss: %.3f | Accuracy: %.3f | Performance: %.3f' % (        
        loss,
        accuracy,
        performance
    ))

def compute_performace(accuracy, loss):
    return 0.7 * loss + 0.3 * accuracy