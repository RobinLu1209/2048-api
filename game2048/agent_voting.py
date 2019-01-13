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


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=True):
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



class CNN_voting(Agent):
    def __init__(self, model1, model2, model3, game, display=None):
        self.game =game
        self.display=display
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

    def step(self):
        present_board = self.game.board
        board = present_board.reshape(1, 1, 4, 4)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = torch.FloatTensor(board)
        board = Variable(board)
        self.search_func1 = self.model1
        self.search_func2 = self.model2
        self.search_func3 = self.model3
        self.search_func1.eval()
        self.search_func2.eval()
        self.search_func3.eval()
        output1 = self.search_func1(board)
        output2 = self.search_func2(board)
        output3 = self.search_func3(board)
        direction1 = output1.data.max(1, keepdim=True)[1].item()
        direction2 = output2.data.max(1, keepdim=True)[1].item()
        direction3 = output3.data.max(1, keepdim=True)[1].item()
        if direction1 == direction2:
            direction = direction1
        elif direction1 == direction3:
            direction = direction3
        elif direction2 == direction3:
            direction = direction2
        else:
            direction = direction1

        return int(direction)