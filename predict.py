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
import json


displaying_choices = ['most', 'top5', 'name']
parser = argparse.ArgumentParser(description='image classifer project')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to json file')
parser.add_argument('--device', type=bool, default=False, help='choose the traning device')
parser.add_argument('--display', type=str,  default='most',choices=displaying_choices, help='choose displaying')
parser.add_argument('--dir', type=str, default='flowers/test/91/image_04890.jpg',help='path to folder of images')
parser.add_argument('--resume', default='flower_checkpoint.pth', type=str, help='path to latest checkpoint')

global args
args = parser.parse_args()


def load_checkpoint(args):
    checkpoint = torch.load(args.resume)
    
    if checkpoint['arch'] == 'densenet':
        model = models.densenet121()
    elif checkpoint['arch'] == 'vgg':
        model = models.vgg16()

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
   
    model.classifier = Network(1024, 102, checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    if args.device:
       if torch.cuda.is_available():
           model.cuda()
       else:
           print('GPU is not available')    
    return model, class_to_idx, idx_to_class

def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    img_pil = Image.open(img_path)

    # Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # preprocess the image
    img_tensor = preprocess(img_pil)

    return img_tensor


def predict(img_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img_tensor = process_image(img_path)
    img_tensor.unsqueeze_(0)
    data = Variable(img_tensor, volatile=True)
    model = model.eval()
    output = model(data)
    # predict the class from an image file

    top5_prob, top5_label = torch.topk(output, 5)

    return top5_prob.exp(), top5_label
model,class_to_idx, idx_to_class=load_checkpoint(args)


img_path=args.dir
img_tensor=process_image(img_path)
x1 = predict(img_path,model)[0]
y1= predict(img_path,model)[1]
x1=x1.detach().numpy()
y1=y1.detach().numpy()
x2=x1.flatten().tolist()
y2=y1.flatten().tolist()

    
if args.display=='most':
    print (x2[0], y2[0])
elif args.display=='top5':
    print (x2,y2)
elif args.display=='name':
    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    cat=idx_to_class[int(y2[0])]
    name1=cat_to_name[str(cat)]
    name1=str(name1)
    cat = [idx_to_class[int(y2[i])] for i in range(5)]
    names=[cat_to_name[str(cat[i])] for i in range(5)]
    print (names[0])



