import random
import numpy as np
from .agent import Agent
from boardgame2 import EMPTY

class RandomChoose(Agent):
    def __init__(self, *args, **kwargs):
        pass

    def decide(self, observation):
        board, player = observation
        return random.choice(np.argwhere(board == EMPTY))