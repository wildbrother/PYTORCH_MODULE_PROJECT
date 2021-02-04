#VGG 구현부
from torchvision.datasets import MNIST
from Modules.VGG_ import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

training_epoch = 100

train_data = MNIST(root="./", train = True, transform = transforms.ToTensor(), download= True)
test_data = MNIST(root="./",train = False, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset= train_data,batch_size= 32, shuffle= True, num_workers=2,drop_last = True)
test_loader = DataLoader(dataset= test_data, batch_size = 32, shuffle=False,num_workers=2,drop_last = True)

features = make_layers(cfg=cfgs['MNIST'],batch_norm=True)
model = VGG(features = features, num_classes = 1000, init_layers= True)

optimizer = optim.Adam(params=model.parameters(),lr=0.001)
loss_func = nn.CrossEntropyLoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


for epoch in range(training_epoch):
    for j,[image,label] in enumerate(train_loader) :
        print(j)
        x = image.to(device)
        y = label.to(device)

        #print(x.shape)

        optimizer.zero_grad()
        out = model.forward(x)
        loss = loss_func(out,y)
        loss.backward()
        optimizer.step()

        if j % 10 == 0 :
            print(loss)
import torchvision.models.googlenet










