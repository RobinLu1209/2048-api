import numpy as np
from sklearn.externals import joblib

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
