import gym
import numpy as np
from collections import namedtuple
from ai import EvaluateAI

class GomukuEnv():
    def __init__(self, board_size=15):
        self.env = gym.make('Gomuku-v0', board_shape=board_size, target_length=5)
        self.opponent = EvaluateAI(board_size)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            return observation, reward, done, info
        board, player = observation
        board_ = board.copy()
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board_[i][j] == -1:
                    board_[i][j] = 2
                elif board_[i][j] == 1:
                    board_[i][j] = 1
        turn = namedtuple("turn", "value")
        turn.value = int(-0.5 * player + 1.5) 
        opponent_y, opponent_x = self.opponent.findBestChess(board_, turn)
        print("opponent_x: ", opponent_x, "opponent_y: ", opponent_y)
        return self.env.step(np.array([opponent_x, opponent_y], dtype=np.int8))

    def close(self):
        self.env.close()

    def is_valid(self, state, action):
        return self.env.is_valid(state, action)

    def render(self):
        self.env.render()

    @property
    def action_space(self):
        return self.env.action_space