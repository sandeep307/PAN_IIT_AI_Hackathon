from __future__ import print_function, division
import sys
sys.path.append('../')
import os
import os.path
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime

import torch.utils.data as data
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

varlr = False
arch = 'inception_v3'
arch_model = models.inception_v3

batch_size_num = 32
learning_rate = 0.001
momentum = 0.9
step_size = 2
num_epochs = 12

train_file_name = '../data/train_aug_8k.txt' # train_98p
test_file_name = '../data/valid.txt'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class Read_Dataset():
    
    def __init__(self, file_path,transform=None):
        self.data = pd.read_csv(file_path, header = None, sep = ' ')
        self.label = self.data.iloc[:, 1].tolist()
        self.img_path = self.data.iloc[:, 0].tolist()   
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.label[index]
        img_path = self.img_path[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        return img_path, image, label



train_val_list = ['train' , 'val']
path_to_input_file = {
    'train' : train_file_name ,
    'val':  test_file_name
    }

path_to_checkpoint = "../models/aug8k_{}_lr_{}".format(arch,learning_rate)
# path_to_checkpoint = os.path.join('../models', path_to_checkpoint)
if not os.path.exists(path_to_checkpoint):
    print("creating directory for checkpoint...")
    os.makedirs(path_to_checkpoint)

image_datasets = {x: Read_Dataset( file_path = path_to_input_file[x], transform = data_transforms[x])
                  for x in train_val_list}

dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size_num, shuffle=True, num_workers=4)
              for x in train_val_list}


print ('Current time and date is: ' + str(datetime.now()) )
print ('train file is: ' + train_file_name )
dataset_sizes = {x: len(image_datasets[x]) for x in train_val_list}
print ('dataset_sizes: ' + str(dataset_sizes) )

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("checking gpu")
print(use_gpu)

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in train_val_list:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, data in enumerate(dataloders[phase]):
                # get the inputs
                iamge_path, inputs, labels = data

                # wrap them in Variable
#                 if use_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device) 

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase=='train': 
                    outputs, aux = model(inputs) 
                    loss = criterion(outputs, labels) + criterion(aux, labels)
                else: 
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]  
            
            ver = torch.__version__
            if '0.3' in ver:
                epoch_acc = running_corrects / dataset_sizes[phase]
                
            else: 
                epoch_acc = running_corrects.cpu().numpy() / dataset_sizes[phase]
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                save_checkpoint(epoch,{
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': epoch_acc,
                    'optimizer' : optimizer.state_dict(),
                })
        epoch_end = time.time() - epoch_start
        print('{}th epoch training completed in {:.0f}m {:.0f}s'.format(epoch,
        epoch_end // 60, epoch_end % 60))
        print()

    time_elapsed = time.time() - since
    print('Entire Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(epoch, state, path_to_checkpoint = path_to_checkpoint, filename = 'batch'+ str(batch_size_num) + '_ckpt.pth.tar'):
    filepath = os.path.join(path_to_checkpoint, "epoch_" + str(epoch) + "_" + filename)
    torch.save(state, filepath)          



# number of classes in train dataset
df = pd.read_csv(path_to_input_file['train'], header = None, sep =' ')
df.columns = ['path', 'label']
n_classes = df['label'].nunique()
print ('number of classes in train dataset:' + str(n_classes) )

# number of classes in test dataset
df = pd.read_csv(path_to_input_file['val'], header = None, sep =' ')
df.columns = ['path', 'label']
n_classes_test = df['label'].nunique()
print ('number of classes in test dataset:' + str(n_classes_test) )

print ('using inception model architecture...')
model_ft = arch_model(pretrained=True)

num_ftrs_aux = model_ft.AuxLogits.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(num_ftrs_aux, n_classes)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, n_classes)
    
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, n_classes)


modelPath = path_to_checkpoint + '/*'
print (modelPath)
list_of_files = glob.glob(modelPath) 

if len(list_of_files) > 0:
    model_file = max(list_of_files, key=os.path.getctime)
    print("picking model file : "+str(model_file))

    checkpoint = torch.load(model_file)
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model_ft.load_state_dict(checkpoint['state_dict'])
    
# model_ft = model_ft.cuda() 
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)

# Decay LR by a factor of 0.1 every 4 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
