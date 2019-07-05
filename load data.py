import torch
from torchvision.transforms import *
from torch.utils.data import *
import torchvision

# Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


import torch
from torchvision.transforms import *
from torch.utils.data import *
import torchvision

# Load the Data
 
# Set train and valid directory paths
train_directory = 'C:\\Users\\pedro.pereira\\Desktop\\Rede Neural\\train'
test_directory = 'C:\\Users\\pedro.pereira\\Desktop\\Rede Neural\\test'
valid_directory = 'C:\\Users\\pedro.pereira\\Desktop\\Rede Neural\\valid'
 
# Batch size
bs = 32
 
# Number of classes
num_classes = 5
 
# Load Data from folders
data = {
    'train': torchvision.datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': torchvision.datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': torchvision.datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
 
# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
 
# Create iterators for the Data loaded using DataLoader module
train_data = torch.utils.data.DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data = torch.utils.data.DataLoader(data['valid'], batch_size=bs, shuffle=True)
test_data = torch.utils.data.DataLoader(data['test'], batch_size=bs, shuffle=True)
 
# Print the train, validation and test set data sizes
train_data_size, valid_data_size, test_data_size