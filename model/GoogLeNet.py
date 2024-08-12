import torch
import torch.nn as nn

def conv_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1,1),
        nn.ReLU()
    )
    return model

def conv_1_3(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1,1),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 3,1,1),
        nn.ReLU()
    )
    return model

def conv_1_5(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1,1),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 5,1,2),
        nn.ReLU()
    )
    return model

def max_1_5(in_dim, pool_dim):
    model = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_dim, pool_dim,1,1),
        nn.ReLU()
    )
    return model

class inception(nn.Module):
    def __init__(self, in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool_dim):
        super(inception, self).__init__()

        self.conv_1 = conv_1(in_dim, out_dim_1)

        self.conv_1_3 = conv_1_3(in_dim, mid_dim_3, out_dim_3)

        self.conv_1_5 = conv_1_5(in_dim, mid_dim_5, out_dim_5)

        self.max_1_5 = max_1_5(in_dim, pool_dim)

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_1_5(x)
        
        output = torch.cat([out_1, out_2, out_3, out_4], 1)
        
        return output
    
class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer_2 = nn.Sequential(
            inception(192,64,96,128,16,32,32),
            inception(256, 128, 128,192,32,96,64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer_3 = nn.Sequential(
            inception(480, 192, 96, 208, 16,48,64),
            inception(512, 160, 112, 224, 24,64,64),
            inception(512, 128,128,256,24,64,64),
            inception(512, 112, 144, 288,32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer_4 = nn.Sequential(
            inception(832, 256,160,320, 32,128, 128),
            inception(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        )

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = x.view(x.size(0),-1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x


    
def googlenet():
    return GoogLeNet()
