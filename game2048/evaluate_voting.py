from game2048.game import Game
from game2048.displays import Display
from sklearn.externals import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

#-------------------------------------------------------------
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
#-------------------------------------------------------------
class deepNet(nn.Module):
    def __init__(self):
        super(deepNet, self).__init__()
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
#-------------------------------------------------------------
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


class Net4096(nn.Module):
    def __init__(self):
        super(Net4096, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.residual = self._make_layer(Resblock,128,4)
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
#-------------------------------------------------------------

def single_run(size, model1,model2,model3, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(model1,model2,model3,game,display=Display(), **kwargs)
    agent.play(verbose=False)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    from game2048.agent_voting import CNN_voting as TestAgent
    '''===================='''
    model1 = Net6()
    model2 = deepNet()
    model3 = Net4096()
    model1.load_state_dict(torch.load('./game2048/CNN+6ResNet+2048big.pkl'))
    model2.load_state_dict(torch.load('./game2048/params_deepCNN_epoch_14.pkl'))
    model3.load_state_dict(torch.load('./game2048/params_4096_epoch_15.pkl'))
    scores = []
    t2 = time.time()
    for _ in range(N_TESTS):
        t1 = time.time()
        score = single_run(GAME_SIZE,model1,model2,model3, SCORE_TO_WIN,AgentClass=TestAgent)
        t = time.time() - t1
        print("Time = ",t)  
        scores.append(score)
    
    total = time.time()-t2
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
    print(scores)
    print("Total time = ",total)
