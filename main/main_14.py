import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#import model
import matplotlib.pyplot as plt
import numpy as np
#%cd ERA-V2-Support
from Model.model_11 import ResNet18
import dataset
import train
import utils

SEED = 1


cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)


torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

train_data = dataset.train()
test_data = dataset.test()
if cuda:
    batch_size = 128
    shuffle = True
    num_workers = 2
    pin_memory = True
    train_loader = dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,train_data)
    test_loader = dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,test_data)
else:
    batch_size =64
    shuffle = True
    num_workers =1
    pin_memory = True
    train_loader = dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,train_data)
    test_loader = dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,test_data)

# functions to show an image
utils.plot_image(train_loader)

device = utils.get_device()
model = ResNet18().to(device)
utils.model_summary(model)

criterion = nn.CrossEntropyLoss()
max_lr, optimizer = utils.get_max_lr(model, train_loader,criterion)

from torch.optim.lr_scheduler import OneCycleLR

EPOCHS = 1

print("Suggested Max LR: ",max_lr)
scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=4/EPOCHS,
        div_factor=10,
        three_phase=False,
        final_div_factor=1,
        anneal_strategy='linear'
    )

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train_acc, train_losses = train.train(model, device, train_loader, optimizer, epoch, scheduler, criterion)
    test_acc, test_losses = train.test(model, device, test_loader, criterion)
    print("LR for current epoch: ",scheduler.get_last_lr())

train.plot_loss_accuracy(train_losses,train_acc,test_losses,test_acc)


misclassified_images,misclassified_labels,misclassified_predictions,prediction = utils.plot_prediction_image(test_loader,device,model)
utils.plot_grad_cam(misclassified_images,misclassified_labels,misclassified_predictions,prediction,model)