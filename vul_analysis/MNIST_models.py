import torch.nn as nn
import torch.nn.functional as F

class NaiveC(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.layer1(x))
        return self.layer2(output)

class NormalC(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 20)
        self.layer4 = nn.Linear(20, 10)
        self.layer5 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.layer1(x))
        h2 = self.relu(self.layer2(h1))
        h3 = self.relu(self.layer3(h2))
        h4 = self.relu(self.layer4(h3))
        return self.layer5(h4)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(3*3*32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (24, 24, 16)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # (20, 20, 16) -> (10, 10, 16)
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) # (6, 6, 32) -> (3, 3, 32)
        x = x.view(-1, 3*3*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class robustified_FC(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 20)
        self.layer4 = nn.Linear(20, 10)
        self.layer5 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

        self.dropout_rate = None
        if dropout_rate is None:
            self.dropout_rate = 0.2
        else: 
            self.dropout_rate = dropout_rate

    def forward(self, x):
        h1 = self.relu(self.layer1(x))
        h2 = self.relu(self.layer2(h1))

        h2 = F.dropout(h2, p=self.dropout_rate) # robustifying layer 

        h3 = self.relu(self.layer3(h2))
        h4 = self.relu(self.layer4(h3))
        return self.layer5(h4)

class robustified_CNN(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(3*3*32, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout_rate = None
        if dropout_rate is None:
            self.dropout_rate = 0.2
            print('dropout rate not specified, create robustified CNN with dropout rate 0.2')
        else: 
            self.dropout_rate = dropout_rate
            print('create robustified CNN with dropout rate', self.dropout_rate)



    def forward(self, x):
        x = F.relu(self.conv1(x)) # (24, 24, 16)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # (20, 20, 16) -> (10, 10, 16)
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) # (6, 6, 32) -> (3, 3, 32)
        x = x.view(-1, 3*3*32)

        x = F.dropout(x, p=self.dropout_rate) # This statement is combined with LP_utils/extract_all_LP

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    