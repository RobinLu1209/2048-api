import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv

batch_size = 256
NUM_EPOCHS = 20

#------------------读入数据-----------------------------
csv_data = pd.read_csv('2048Train6.csv')
csv_data = csv_data.values
board_data = csv_data[:,0:16]
X = np.int64(board_data)
X = np.reshape(X, (-1,4,4))
direction_data = csv_data[:,16]
Y = np.int64(direction_data)
#-------------------------------------------------------

writer = SummaryWriter('./runs/batch_size_{}_epoch{}'.format(batch_size, NUM_EPOCHS))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=False)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

train_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False
)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(4, 4), padding=(2, 2))
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(5, 5), padding=(2, 2))
        self.conv8 = nn.Conv2d(512, 512, kernel_size=(5, 5), padding=(2, 2))

        self.fc1 = nn.Linear(512 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(512 * 6 * 6)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)

        self.initialize()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = x.view(-1, 512 * 6 * 6)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.fc3(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')


if __name__ == '__main__':
    inputs = torch.ones(16, 1, 4, 4)
    model = Net()
    outputs = model(inputs)

model = Net()
model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=[0, 1])
optimizer = optim.Adam(model.parameters(), lr = 0.001)


def train(epoch):
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        data = data.unsqueeze(dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    torch.save(model.state_dict(), 'model_saved/params_deepCNN_epoch_{}.pkl'.format(epoch))


def test(epoch):
    test_loss = 0
    correct = 0
    # model.load_state_dict(torch.load('model_saved/params_epoch_{}.pkl'.format(epoch)))
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        data = data.unsqueeze(dim=1)
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    writer.add_scalar('Test_loss', test_loss, epoch)
    writer.add_scalar('Test_accuracy', 100. * float(correct) / len(test_loader.dataset), epoch)


for epoch in range(0, NUM_EPOCHS):
    train(epoch)
    test(epoch)
