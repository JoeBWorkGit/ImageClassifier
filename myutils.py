import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models, transforms, utils
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import json
import time
import copy
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train = transforms.Compose([transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])])
    valid = transforms.Compose([transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])])
    test = transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])

        
    train_data = datasets.ImageFolder(train_dir, transform=train)
    validation_data = datasets.ImageFolder(valid_dir, transform=valid)
    test_data = datasets.ImageFolder(test_dir ,transform = test)
   
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    

    print('Data Loaded')
    return trainloader, validloader, testloader, train_data
  
def build_the_model(arch, hidden_units,learning_rate,gpu):

    if arch.lower() == "vgg16":
        model = models.vgg16(pretrained=True)
        print('model vgg16 will be built')
    else:
        model = models.vgg19(pretrained=True)
        print('model vgg19 (default) will be built')

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print(f"Model bas been built using {arch} with {hidden_units} hidden units.")

    return model, criterion, optimizer

def train_model(model, epochs,trainloader, validloader, criterion, optimizer,gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu")
    model.to(device)
    steps = 0
    print_every = 20
    since = time.time()
    
    print('Start training model')
    for e in range(epochs):
      
        running_loss = 0

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
           
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Carrying out validation step
            if steps % print_every == 0:
               
                model.eval()
               
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion,device)
             
                print(f"Epoch {e+1}/{epochs}\
                Training Loss: {round(running_loss/print_every,3)}\
                Valid Loss: {round(valid_loss/len(validloader),3)}\
                Valid Accuracy: {round(float(accuracy/len(validloader)),3)}")
                
                time_elapsed = time.time() - since
                print('Elapsed training time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

                running_loss = 0
                # Turning training back on
                model.train()
                
    time_elapsed = time.time() - since
    print('Total training time {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    return model, optimizer

def validation(model, dataloader, criterion, device):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in iter(dataloader):
            
            images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU
            
            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy


def save_model(model, train_data, optimizer, save_dir, arch, epochs):
    model.class_to_idx = train_data.class_to_idx
   # model.class_to_idx = trainloader.class_to_idx
    model.cpu()  
    checkpoint = {'arch': arch,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}

    torch.save(checkpoint, save_dir)
    return

def load_checkpoint(arch):
    if arch.lower() == "vgg19":
       model = models.vgg19(pretrained=True)
    else:
       model = models.vgg16(pretrained=True)
    
    classifier = nn.Sequential(OrderedDict([
                            ('dropout1', nn.Dropout(0.1)),
                            ('fc1', nn.Linear(25088,1024)), # 25088 must match
                            ('relu1', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.1)),
                            ('fc2', nn.Linear(1024, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier
    checkpoint = torch.load('checkpoint.pth')
    #  checkpoint = torch.load('checkpoint.pth', map_location='cpu')
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    print('model loaded')
    return model

def predict(image_path, model, topk, gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu")   
    # device = torch.device("cuda:0")
    model.eval()
    model.to(device)
   

    processed_image = process_image(image_path)
    print('image processed')
    img_tensor = torch.from_numpy(processed_image).type(torch.cuda.FloatTensor)
    img_add_dim = img_tensor.unsqueeze_(0)

    with torch.no_grad():
        output = model.forward(img_add_dim)

   
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
   
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    class_to_idx = model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list

def process_image(image_path):
   
    # TODO: Process a PIL image for use in a PyTorch model
    pimage = Image.open(image_path)
   
    if pimage.size[0] > pimage.size[1]:
        pimage.thumbnail((10000, 256))
    else:
        pimage.thumbnail((256, 10000))
 
    left_margin = (pimage.width-224)/2
    bottom_margin = (pimage.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    pimage = pimage.crop((left_margin, bottom_margin, right_margin,   
                       top_margin))
   
    pimage = np.array(pimage)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    pimage = (pimage - mean)/std
    
    pimage = pimage.transpose((2, 0, 1))
    
    return pimage