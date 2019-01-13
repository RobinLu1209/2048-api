import numpy as np
from sklearn.externals import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import time
import pandas as pd
import csv

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



class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction



class deepCNN(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.search_func = Net()
        self.search_func.load_state_dict(torch.load('./game2048/params_deepCNN_epoch_14.pkl'))

    def step(self):
        self.search_func.eval()
        present_board = self.game.board
        board = present_board.reshape(1, 1, 4, 4)
        #board = np.int64(board)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = torch.FloatTensor(board)
        board = Variable(board)
        output = self.search_func(board)
        direction = output.data.max(1, keepdim=True)[1].item()
        return int(direction)

