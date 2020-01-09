
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import copy
import os

data_trans = {
             'train': transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ColorJitter(contrast = 0),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
             'test': transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

path = 'datasets/chest_xray'

img_dataset = { x : datasets.ImageFolder(os.path.join(path, x),
                                   transform = data_trans[x])
               for x in ['train', 'test']}

dataloader = { x : torch.utils.data.DataLoader(img_dataset[x], 
                                              batch_size = 24,
                                              shuffle = True,
                                              num_workers = 4) 
              for x in ['train','test']}

dataset_sizes = { x : len(img_dataset[x]) for x in ['train', 'test']}
class_names = img_dataset['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Visualize few sample training images.

def imshow(img, title):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (std * img) + mean 
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if type(title) == list:
        plt.title(title)
    elif title is not None:
        plt.title('Ground: ' + title['G'] + '\n' + 'Predicted: ' + title['P'] + '\n')
    
images, labels = next(iter(dataloader['train']))
title = [class_names[i] for i in labels]
imshow(torchvision.utils.make_grid(images), title)

# Training a Model. 
def training_model(model, criterion, optimizer, scheduler, epochs = 30):
    tic = time.time()
    best_model_param = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0 
    
    for epoch in range(epochs):
        print('{} / {}'.format(epoch + 1, epochs))
        print('-' * 20)
        
        for mode in ['train', 'test']:
            if mode == 'train':
                model.train()
            else: 
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in dataloader[mode]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += (loss.item() * inputs.size(0)) 
                running_corrects += torch.sum(predictions == labels.data)
            
            if mode == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[mode]    
            epoch_acc = running_corrects / dataset_sizes[mode]
            
            print('{} loss : {} acc : {}'.format(mode, epoch_loss, epoch_acc))
            
            if mode == 'test' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_param = copy.deepcopy(model.state_dict())
            
        print()
        
    toc = time.time()
    elapsed_time = toc - tic
    print('Training completed in {}min {}sec'.format(elapsed_time // 60, elapsed_time % 60))
    print('Best Validation Accuracy : {}'.format(best_accuracy))
    model.load_state_dict(best_model_param)
    return model

#######################################################################################

# Convnet Fine Tuning.
model_ft = models.resnet50(pretrained = True)
num_feat = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_feat, 2)


model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.01, momentum = 0.9)
#optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.01, betas=(0.9, 0.999))

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 2.0)

model_ft = training_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, epochs = 30)
torch.save(model_ft.state_dict(), os.path.join(path, 'model_FT.pth'))         

#######################################################################################

# Convnet Fixed Feature Extractor.
model_fft = torchvision.models.resnet50(pretrained = True)

for para in model_fft.parameters():
    para.require_grad = False

num1_feat = model_fft.fc.in_features
model_fft.fc = nn.Linear(num1_feat, 2)
model_fft = model_fft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_fft = optim.SGD(model_fft.fc.parameters(), lr = 0.01, momentum = 0.9)
exp_lr_scheduler_fft = lr_scheduler.StepLR(optimizer_fft, step_size=7, gamma = 2.0)    

model_fft = training_model(model_fft, criterion, optimizer_fft,
                            exp_lr_scheduler_fft, epochs = 30)
torch.save(model_fft.state_dict(), os.path.join(path, 'model_FFT.pth'))  

######################################################################################

# Prediction done on sample images.

sample_data = torchvision.datasets.ImageFolder('datasets/chest_xray/val', 
                                               transform = data_trans['test'])

sample_loader = torch.utils.data.DataLoader(sample_data, batch_size = 4,
                                            shuffle = True, num_workers = 4)

dataiter = iter(sample_loader)
images, labels = next(dataiter) 

images = images.to(device)
labels = labels.to(device)

sample_model = models.resnet50(pretrained = True)
for para in sample_model.parameters():
    para.require_grad = False

features = sample_model.fc.in_features
sample_model.fc = nn.Linear(features, 2)
sample_model = sample_model.to(device)

sample_model.load_state_dict(torch.load('datasets/chest_xray/model_FFT.pth'))


sample_output = sample_model(images)
_, sample_pred = torch.max(sample_output, 1)
title = {'P' : ' '.join(class_names[sample_pred[j]] for j in range(4)),
         'G' : ' '.join(class_names[labels[j]] for j in range(4))}

%matplotlib auto
imshow(torchvision.utils.make_grid(images.cpu()), title)


   
    
