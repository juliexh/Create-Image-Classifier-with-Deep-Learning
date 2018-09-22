from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib

import matplotlib.pyplot as plt
import warnings

import time
import os
import copy
from collections import OrderedDict
import torchvision.models as models
densenet121 = models.densenet121(pretrained=True)

vgg16 = models.vgg16(pretrained=True)


model_names = {'densenet': densenet121, 'vgg': vgg16}
# Creates parse
parser = argparse.ArgumentParser(description='image classifer project')

    # Creates 3 command line arguments args.dir for path to images files,
    # args.arch which CNN model to use for classification, args.labels path to
    # text file with names of dogs.
parser.add_argument('--dir', type=str, default='flowers', help='path to folder of images')    
parser.add_argument('--path', type=str,  default='flower_checkpoint.pth',help='path to save the model')
parser.add_argument('--device', type=bool, default=False, help='choose the traning device')
parser.add_argument('--arch', type=str, default='densenet',choices=model_names,help='chosen model')
parser.add_argument('--epochs', default=2, type=int,help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,help='initial learning rate')
parser.add_argument('--hidden_units', type=int, default=[500, 200], help='control the number of layers and hidden units')

global args
args = parser.parse_args()



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = args.dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'valid', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

class_to_idx = image_datasets['train'].class_to_idx


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
#                inputs = inputs.to(device)
#                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    m = nn.LogSoftmax(dim=1)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(m(outputs), labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
if args.arch=='vgg':
    model=vgg16
else:
    model=densenet121
for param in model.parameters():
    param.requires_grad = False
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
model.classifier = Network(1024, 102, args.hidden_units)
    

print (model)
if args.device:
   if torch.cuda.is_available():
       model.cuda()
   else:
       print('GPU is not available')

criterion = nn.NLLLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer,exp_lr_scheduler,num_epochs=args.epochs)
checkpoint = {'arch': args.arch,
             'state_dict': model.state_dict(),
             'optimizer':optimizer.state_dict(),
             'epoch':args.epochs,
             'class_to_idx':image_datasets['train'].class_to_idx,
             'hidden_units': args.hidden_units,
             'lr': args.lr}
torch.save(checkpoint, args.path)