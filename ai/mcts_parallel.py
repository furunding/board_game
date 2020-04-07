from boardgame2 import BLACK, WHITE, EMPTY
from boardgame2 import strfboard

import numpy as np
import random
import collections
from .agent import Agent

from multiprocessing import Process, Pool, JoinableQueue, cpu_count
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

class MCTSParallelPlayer(Agent):
    def __init__(self, env, sim_count=16, trade_off=3, neighbor_range=1, process_num=0):
        self.sim_count = sim_count
        self.neighborhood = lambda loc: itertools.product(range(loc[0]-neighbor_range, loc[0]+neighbor_range+1), range(loc[1]-neighbor_range, loc[1]+neighbor_range+1))
        self.env = env
        self.trade_off = trade_off
        self.process_num = process_num if process_num else cpu_count()
        self.parallel()
        self.reset()

    def parallel(self):
        self.task_queue = JoinableQueue(1)
        self.result_queue = JoinableQueue(self.process_num)
        self.workers = [Process(target=self.rollout_worker, args=(self.env, self.task_queue, self.result_queue)) for _ in range(self.process_num)]
        for worker in self.workers:
            worker.start()
        self.update_thread = Thread(target=self.update_statics)
        self.update_thread.start()

    def reset(self):
        self.count = KeyHashDefaultDict(int)
        self.uob = KeyHashDefaultDict(int)
        self.win = KeyHashDefaultDict(int)
        self.child_info = KeyHashDefaultDict(list)
        self.father_info = KeyHashDefaultDict()

    def get_winner(self, state):
        return self.env.get_winner(state)

    def hash_convert(self, state):
        board, player = state
        return (strfboard(board), player)
        
    def simulation(self, state):
        leaf_state = self.select(state)
        if self.get_winner(leaf_state) != None:
            self.back_propagate(leaf_state, self.get_winner(leaf_state))
        else:
            if self.count[leaf_state] + self.uob[leaf_state]== 0:
                self.rollout(leaf_state)     
            else:
                self.expand(leaf_state)
                next_state = random.choice(self.child_info[leaf_state])
                if self.get_winner(next_state) != None:
                    self.back_propagate(next_state, self.get_winner(next_state))
                else:
                    self.rollout(next_state)

    def select(self, state):
        cur_state = state
        while cur_state in self.child_info:
            self.uob[cur_state] += 1
            childs = self.child_info[cur_state]
            unexplored_childs = [child for child in childs if self.count[child]+self.uob[child]==0]
            if unexplored_childs:
                cur_state = random.choice(unexplored_childs)
            else:
                ucbs = [self.win[next_state] / (self.count[next_state]+epision) + np.sqrt(self.trade_off * np.log(self.count[cur_state]+self.uob[cur_state]) / (self.count[next_state]+self.uob[next_state])) for next_state in childs]
                max_ucb = max(ucbs)
                best_childs = [child for child, ucb in zip(childs, ucbs) if ucb == max_ucb]
                cur_state = random.choice(best_childs)
        return cur_state

    def expand(self, state):
        board, player = state
        cur_state = state
        self.uob[cur_state] += 1
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

    def expand_worker(self, state):
        pass
    
    def rollout(self, state):
        self.uob[state] += 1
        pre_actions = []
        cur_state = state
        while self.father_info[cur_state]:
            pre_actions.append(np.argwhere(self.father_info[cur_state][0] - cur_state[0] != EMPTY)[0])
            cur_state = self.father_info[cur_state]
        self.task_queue.put({"mode": "rollout", "node": state, "pre_actions": pre_actions}, block=True)

    @staticmethod
    def rollout_worker(env, task_q, result_q):
        trans = Transfer()
        while True:
            data = task_q.get(True)
            state = data["node"]
            pre_actions = data["pre_actions"]
            board, player = state
            while True:
                vaild_actions = list(zip(*np.where(board == 0)))
                if vaild_actions:
                    # random_action = random.choice(vaild_actions)
                    action = trans.decide(pre_actions)
                    (board, player), winner, done, info = env.next_step((board, player), action)
                else:
                    result_q.put((state, random.choice([WHITE, BLACK])), block=True)
                    task_q.task_done()
                    break
                if done:
                    result_q.put((state, winner), block=True)
                    task_q.task_done()
                    break

    def back_propagate(self, state, winner):
        cur_state = state
        while cur_state != None:
            self.uob[cur_state] -= 1
            self.count[cur_state] += 1
            if winner == cur_state[1]: self.win[cur_state] += 1
            cur_state = self.father_info[cur_state]

    def update_statics(self):
        while True:
            data = self.result_queue.get(True)
            self.back_propagate(*data)
            self.result_queue.task_done()

    def closeout(self):
        self.task_queue.join()
        self.result_queue.join()

    def decide(self, state):
        board, player = state
        self.father_info[state] = None
        # while self.count[state] < self.sim_count:
        for i in range(self.sim_count):
            self.simulation(state)
        self.closeout()
        childs = self.child_info[state]
        win_rates = [self.win[next_state] / (self.count[next_state]+epision) for next_state in childs]
        max_win_rate = max(win_rates)
        best_childs = [child for child, win_rate in zip(childs, win_rates) if win_rate == max_win_rate]
        best_child = random.choice(best_childs)
        return np.argwhere(best_child[0] - board)[0]