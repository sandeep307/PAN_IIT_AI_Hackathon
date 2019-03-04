import os, glob
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import Softmax
os.environ["CUDA_VISIBLE_DEVICES"]="2" 
from torch import nn

test_file_name = '../data/test.txt'
batch_size = 256
num_classes = 6

# model_path = '../models/perc98_inception_v3_lr_0.001' + '/*' all_data_res34_lr_0.05   #
model_path = '../models/aug8k_inception_v3_lr_0.001' + '/*'
arch = 'inception_v3'
arch_model = models.inception_v3
output_folder = '../submissions'


class Read_Dataset():
    
    def __init__(self, file_path,transform=None):
        self.data = pd.read_csv(file_path, header = None, sep = ' ')
        self.img_path = self.data.iloc[:, 0].tolist()   
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.img_path[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        return img_path, image
    
    
def get_dataloader(test_file_name, batch_size=64):

    image_datasets = Read_Dataset( file_path = test_file_name, transform = data_transforms)

    dataloader = DataLoader(image_datasets, batch_size = batch_size, shuffle=False, num_workers=4)

    print ('dataset_size: {}'.format( len(image_datasets) ) )
    return dataloader

    
    
def get_model(model_path, arch_model, use_gpu = True):

    list_of_files = glob.glob(model_path) 
    model_file = max(list_of_files, key=os.path.getctime)
    print('path {} and model file {}'.format(model_path, model_file))

    model_ft = arch_model(pretrained=True)
    
    if arch_model == models.densenet121:
        model_ft.classifier = torch.nn.Linear(model_ft.classifier.in_features, num_classes)
    
    elif arch_model == models.inception_v3:
        num_ftrs_aux = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    else:
        model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)

    checkpoint = torch.load(model_file)
    model_ft.load_state_dict(checkpoint['state_dict'])
    if use_gpu:
        model_ft = model_ft.cuda()
    return model_ft


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
       
    
def get_predictions(model, dataloaders, use_gpu = True):
    
    model_ft.eval()
    results = pd.DataFrame()
    with torch.no_grad():
        
        for i, data in tqdm(enumerate(dataloaders)):
            
            path, inputs = data
            if use_gpu:
                inputs = inputs.cuda()
                
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs.data, 1)
#             outputs, aux = model(inputs)
            
#             preds = outputs.data.cpu().numpy()
            for j in range(outputs.size()[0]):
#                 label_name = preds[j].argsort()[-3:][::-1]
                label_name = int( preds[j].cpu().numpy() ) + 1
                prob = softmax (outputs.data[j].cpu().numpy())
                temp = pd.DataFrame({'path': [path[j]], 'category' : [label_name], 'Probablity' : [prob] })
                results = results.append(temp)
                
    return results        
        
    
if __name__ =="__main__":

    if arch_model == models.inception_v3:
        data_transforms = transforms.Compose([
            transforms.Resize( (299,299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    
    else:
        data_transforms = transforms.Compose([
            transforms.Resize( (224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            

    #getting dataloader
    dataloader = get_dataloader(test_file_name, batch_size)
    
    #loading the model
    model_ft = get_model(model_path, arch_model)
    
    submission = get_predictions(model_ft, dataloader)
    submission['id'] = [x.split('/')[-1].split('.')[0] for x in submission.path]
    
    #writing probablity file 
    results_prob = submission[['id', 'category', 'Probablity' ]]

    
    filename = f'aug8k_{arch}_prob_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    results_prob.to_csv(f'{output_folder}/{filename}', index=False)
        
    submission = submission[['id', 'category']]
#     submission.Category = [ ','.join([str(x) for x in list(y)]) for y in submission.category.tolist()]
    
    filename = f'aug8k_{arch}_submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    submission.to_csv(f'{output_folder}/{filename}', index=False)
                           