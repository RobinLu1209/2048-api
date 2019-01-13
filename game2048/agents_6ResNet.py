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


class Resblock(nn.Module):
    def __init__(self, channel_num):
        super(Resblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channel_num),
            nn.ReLU(),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channel_num)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        return out


class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.residual = self._make_layer(Resblock,128,6)
        self.fc1 = nn.Linear(128 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(128 * 5 * 5)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)

        self.initialize()

    def _make_layer(self, block , channel_num , res_num):
        layers = []
        for i in range(res_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.residual(x)
        x = x.view(-1, 128 * 5 * 5)
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


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.residual = self._make_layer(Resblock,128,6)
        self.fc1 = nn.Linear(128 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(128 * 5 * 5)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)

        self.initialize()

    def _make_layer(self, block , channel_num , res_num):
        layers = []
        for i in range(res_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.residual(x)
        x = x.view(-1, 128 * 5 * 5)
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


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class LDA(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)


        clf2 = joblib.load('LDA_model.pkl')
        self.search_func = clf2.predict
        # print(clf2.predict(board))

    def step(self):
        present_board = self.game.board
        board = present_board.reshape(1, 16)
        board = np.int32(board)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = np.int32(board)
        board = board / 11.0
        direction = self.search_func(board)
        return int(direction)

class CNN4096(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.search_func = Net4()
        self.search_func.load_state_dict(torch.load('./game2048/params_4096_epoch_26.pkl'))

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


class CNN_ResNet6(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.search_func = Net6()
        self.search_func.load_state_dict(torch.load('./game2048/CNN+6ResNet+2048big.pkl'))

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





class KNN(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)


        clf4 = joblib.load('KNN_model.pkl')
        self.search_func = clf4.predict

    def step(self):
        present_board = self.game.board
        board = present_board.reshape(1, 16)
        board = np.int32(board)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = np.int32(board)
        board = board / 11.0
        direction = self.search_func(board)
        return int(direction)
