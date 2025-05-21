import os
import torchvision
from torchvision.transforms import transforms
import torch.optim
import torch.nn as nn
from cnn import CNN
import sys
from datetime import date
import logging
from utils.utils import create_dir, get_accuracy

sys.path.append('../')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir())))

# Set training parameters
epochs = 250
batch_size = 16
lr = 0.01
workers = 2

def main():

    # Create training session
    today = date.today()
    today = str(today.strftime('%m-%d-%Y'))
    dir_ = str(root_dir + '/models/CNN/train-' + today)
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
    train_data = torchvision.datasets.ImageFolder(root=root_dir + '/data/plots/train/',
                                                  transform=transforms['train'])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    model = CNN()
    model = model.to(device)

    loss_fx = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_steps = 0

        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)

            prediction = model(img)
            loss = loss_fx(prediction, label)
            epoch_accuracy = get_accuracy(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        # Training status

        print('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (
            epoch+1,
            epoch,
            epoch_loss,
            epoch_accuracy
        ))

        logging.info('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (
            epoch+1,
            epoch,
            epoch_loss,
            epoch_accuracy
        ))

        # Save Model
        model_file_name = 'CNN-' + today + '.pt'
        torch.save(model.state_dict(), os.path.join(dir_, model_file_name))

if __name__ == '__main__':
    main()