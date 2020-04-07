from boardgame2 import BLACK, WHITE, EMPTY
from boardgame2 import strfboard

import numpy as np
import random
import collections
from .agent import Agent

from multiprocessing import Process, Pool, JoinableQueue
from threading import Thread
import time
from copy import deepcopy
from functools import reduce
import operator
import itertools

from .windows_gobang.new_trans import Transfer

epision = 1e-6
# unexplored_ub = 1000

class LengthLimitedQueue(collections.deque):
    def __init__(self, max_len):
        self.max_len = max_len
    
    def is_full(self):
        return len(self) == self.max_len

    def is_empty(self):
        return len(self) == 0

    def push(self, item):
        self.appendleft(item)

class KeyHashDefaultDict(collections.defaultdict):
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, state):
        board, player = state
        board_ = board if isinstance(board, str) else strfboard(board)
        return super().__getitem__((board_, player))

    def __setitem__(self, state, value):
        board, player = state
        board_ = board if isinstance(board, str) else strfboard(board)
        return super().__setitem__((board_, player), value)

    def __contains__(self, state):
        board, player = state
        board_ = board if isinstance(board, str) else strfboard(board)
        return super().__contains__((board_, player))

def net_policy_fn(board):
    pass

def random_policy_fn(board):
    pass


class MCTSPlayer(Agent):
    def __init__(self, env, sim_count=100, trade_off=2, neighbor_range=1):
        self.sim_count = sim_count
        self.env = env
        self.trade_off = trade_off
        self.neighborhood = lambda loc: itertools.product(range(loc[0]-neighbor_range, loc[0]+neighbor_range+1), range(loc[1]-neighbor_range, loc[1]+neighbor_range+1))
        self.move_policy = Transfer()
        self.reset()

    def reset(self):
        self.count = KeyHashDefaultDict(int)
        self.count = KeyHashDefaultDict(int)
        self.win = KeyHashDefaultDict(int)
        self.child_info = KeyHashDefaultDict(list)
        self.father_info = KeyHashDefaultDict()

    def get_winner(self, state):
        board, player = state
        return self.env.get_winner((board, player))

    def hash_convert(self, state):
        board, player = state
        return (strfboard(board), player)
        
    def simulation(self, state):
        leaf_state = self.select(state)
        if self.get_winner(leaf_state) != None:
            self.back_propagate(leaf_state, self.get_winner(leaf_state))
        else:
            if self.count[leaf_state] == 0:
                self.back_propagate(*self.rollout(leaf_state))    
            else:
                self.expand(leaf_state)
                next_state = random.choice(self.child_info[leaf_state])
                if self.get_winner(next_state) != None:
                    self.back_propagate(next_state, self.get_winner(next_state))
                else:
                    self.back_propagate(*self.rollout(next_state))

    def select(self, state):
        cur_state = state
        while cur_state in self.child_info:
            self.count[cur_state] += 1
            childs = self.child_info[cur_state]
            ucbs = [self.win[next_state] / (self.count[next_state]+epision) + np.sqrt(self.trade_off * np.log(self.count[cur_state]) / (self.count[next_state]+epision)) for next_state in childs]
            max_ucb = max(ucbs)
            best_childs = [child for child, ucb in zip(childs, ucbs) if ucb == max_ucb]
            cur_state = random.choice(best_childs)
        return cur_state

    def expand(self, state):
        board, player = state
        cur_state = state
        self.count[cur_state] += 1
        if np.any(board != EMPTY):
            neighbor_valid_actions = reduce(operator.concat, [[act for act in self.neighborhood(action) if 0<=act[0]<board.shape[0] and 0<=act[1]<board.shape[1] and board[act] == EMPTY] for action in np.argwhere(board != EMPTY)])
        else:
            neighbor_valid_actions = [(np.array(board.shape) / 2).astype(np.int8)]
        
        for action in neighbor_valid_actions:
            next_state, winner, done, _ = self.env.next_step(cur_state, action)
            if done:
                self.child_info[cur_state] = [next_state]
                self.father_info[next_state] = cur_state
                break
            self.child_info[cur_state].append(next_state)
            self.father_info[next_state] = cur_state

    def rollout(self, state):
        self.count[state] += 1
        board, player = state
        while True:
            vaild_actions = list(zip(*np.where(board == 0)))
            if vaild_actions:
                self.move_policy 
                random_action = random.choice(vaild_actions)
                (board, player), winner, done, info = self.env.next_step((board, player), random_action)
            else:
                return state, random.choice([WHITE, BLACK]) 
            if done:
                return state, winner

    def back_propagate(self, state, winner):
        cur_state = state
        while cur_state != None:
            if winner == cur_state[1]: self.win[cur_state] += 1
            cur_state = self.father_info[cur_state]

    def decide(self, state):
        board, player = state
        self.father_info[state] = None
        # while self.count[state] < self.sim_count:
        for i in range(self.sim_count):
            self.simulation(state)
        childs = self.child_info[state]
        win_rates = [self.win[next_state] / (self.count[next_state]+epision) for next_state in childs]
        max_win_rate = max(win_rates)
        best_childs = [child for child, win_rate in zip(childs, win_rates) if win_rate == max_win_rate]
        best_child = random.choice(best_childs)
        return np.argwhere(best_child[0] - board)[0]