import gym
import numpy as np
from collections import namedtuple
import boardgame2
from boardgame2 import BLACK, WHITE

class PlayGomuku():
    def __init__(self, board_size=15, target_length=5, black_agent=None, white_agent=None):
        self.env = gym.make('Gomuku-v0', board_shape=board_size, target_length=target_length)
        self.black_agent = black_agent(self.env)
        self.white_agent = white_agent(self.env.board.shape[0])

    def run(self, episodes=1, black_first=True, verbose=False):
        black_win = 0
        white_win = 0
        for episode in range(1, episodes+1):
            observation = self.env.reset()
            done = False
            turn = BLACK if black_first else WHITE
            for step in range(1, 1000):
                action = self.black_agent.decide(observation) if turn == BLACK else self.white_agent.decide(observation)
                observation, winner, done, info = self.env.step(action)
                if verbose:
                    print("第{}步, {}方落子: {}".format(step, "黑" if turn == BLACK else "白", action))
                    print(boardgame2.strfboard(observation[0]))
                turn *= -1
                if done:
                    break
            if winner == BLACK:
                black_win += 1
                print("回合{}, 黑方胜利!".format(episode))
            else:
                white_win += 1
                print("回合{}, 白方胜利!".format(episode))
            self.env.close()
        print("总计{}回合, 黑方胜率: {}, 白方胜率:{}".format(episode, black_win / episode, white_win / episode))
        