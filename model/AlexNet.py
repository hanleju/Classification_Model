import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(48,128,kernel_size=5,stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(192,192, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(128*6*6, 4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)

        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.softmax(x)
        return x
    
def alexnet():
    return AlexNet