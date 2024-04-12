import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms



class Model_10(nn.Module):
    def __init__(self):
        #dropout_value = 0.05
        super(Model_10, self).__init__()
        # Preplayer 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3 , stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 
        # Layer-1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.resnet1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )


    
        # later 2 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.MaxPool2d(kernel_size=2),            
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
        ) 
        
        # layer 3
        self.convblock4 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.MaxPool2d(kernel_size=2),            
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.resnet2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

        )

        # OUTPUT BLOCK
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=4)
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x+self.resnet1(x) 
        x = self.convblock3(x)
        x = self.convblock4(x)
        x  = x+self.resnet2(x)
        x = self.maxpool(x)        
        x = self.convblock5(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)