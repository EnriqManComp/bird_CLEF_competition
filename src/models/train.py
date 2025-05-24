import os

import torchvision
from torchvision.transforms import transforms
import torch.optim
import torch.nn as nn
from models.cnn import CNN

from datetime import date
import logging

from utils.utils import create_dir
from utils.model_utils import get_accuracy




def train(root_dir, **kwargs):

    # Create training session
    today = date.today()
    today = str(today.strftime('%m-%d-%Y'))
    dir_ = str(root_dir + '/bird_CLEF_competition/models/human_voice/CNN/train-' + today)
    print(root_dir)
    create_dir(dir_)

    log_file_name = 'CNN-' + today + '.log'
    logging.basicConfig(filename=os.path.join(dir_, log_file_name),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        level=logging.INFO
                        )

    # In this project we only use resize and convert to tensor
    # because change the spectogram change the meaning of the features

    transform = {
        'train': transforms.Compose([transforms.Resize([32,32]),
                                     transforms.ToTensor()])
    }

    # Loading training data
    train_data = torchvision.datasets.ImageFolder(root=root_dir + '/bird_CLEF_competition/data/train_images/',
                                                  transform=transform['train'])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=kwargs["workers"]
    )

    model = CNN()
    model = model.to(kwargs["device"])

    loss_fx = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training Loop
    for epoch in range(kwargs["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        best_performance = compute_performace(epoch_accuracy, epoch_loss)
        epoch_steps = 0

        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(kwargs["device"]), label.to(kwargs["device"]).float().view(-1,1)

            prediction = model(img)
            b_prediction = nn.functional.sigmoid(prediction)
            
            loss = loss_fx(b_prediction, label)
            b_prediction = (b_prediction > 0.5).int()

            epoch_accuracy = get_accuracy(b_prediction.squeeze(-1), label.squeeze(-1).int())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        # Training status

        print('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (
            epoch+1,
            kwargs["epochs"],
            epoch_loss,
            epoch_accuracy
        ))

        logging.info('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (
            epoch+1,
            kwargs["epochs"],
            epoch_loss,
            epoch_accuracy
        ))

        scheduler.step()

        performance = compute_performace(epoch_accuracy, epoch_loss)

        if performance > best_performance:
            weights = model.state_dict()
            best_performance = performance
        
    # Save Model
    model_file_name = 'CNN-' + today + '.pt'
    model.load_state_dict(weights)
    torch.save(model.state_dict(), os.path.join(dir_, model_file_name))

def compute_performace(accuracy, loss):
    return 0.7 * loss + 0.3 * accuracy