import numpy as np
import copy
import torch
import os
from torch import nn
from .policy_p_human2 import Policy


class Transfer:

    def __init__(self):
        channel = 8
        self.model_actor = Policy(channel, 15)
        model_path = os.path.join(os.path.dirname(__file__), 'weight-881000.pkl')
        check_point = torch.load(model_path)
        self.model_actor.load_state_dict({k.replace('module.', ''): v for k, v in check_point['net'].items()})
        self.model_actor.eval()
        self.chessList = []
        self.putcount = 0
        pass

    def reset(self):
        # self.chessList.clear()
        del self.chessList[:]

    def put(self, chess):
        self.putcount += 1
        # print('put count', self.putcount)
        self.chessList.append(chess)
        # print('put', self.chessList)
        pass

    def decide(self, pre_actions):
        self.chessList = pre_actions
        return self.chess()

    def chess(self):
        # if len(self.chessList) > 6:
        #6步后开始学习
        features = []
        boardBlack = np.zeros((15, 15))
        boardWhite = np.zeros((15, 15))
        boardEmpty = np.ones((15, 15))
        chess_count = len(self.chessList)
        for si in range(chess_count):
            chess = self.chessList[si]
            ydim = chess[0]
            xdim = chess[1]
            if (si % 2 == 0):
                boardBlack[ydim][xdim] = float(1)
                boardEmpty[ydim][xdim] = float(0)
            else:
                boardWhite[ydim][xdim] = float(1)
                boardEmpty[ydim][xdim] = float(0)
            pass
        features.append(boardBlack)
        features.append(boardWhite)
        features.append(boardEmpty)
        if len(self.chessList) > 4:
            for i in range(1, 6):
                board = np.zeros((15, 15))
                if i == 5:
                    accumlate_chesses = self.chessList[0:1 - i]
                    for chess in accumlate_chesses:
                        ydim = chess[0]
                        xdim = chess[1]
                        board[ydim][xdim] = 1
                else:
                    reverseIndex = 0 - i
                    ch = self.chessList[reverseIndex]
                    ydim = ch[0]
                    xdim = ch[1]
                    board[ydim][xdim] = 1
                features.append(board)
        else:
            count = len(self.chessList)
            for j in range(len(self.chessList)):
                boardless = np.zeros((15, 15))
                reverseIndex = 0 - (count - j)
                ch = self.chessList[reverseIndex]
                boardless = np.zeros((15, 15))
                ydim = ch[0]
                xdim = ch[1]
                boardless[ydim][xdim] = 1
                features.append(boardless)
            remain_c = 5 - count
            for c in range(remain_c):
                b = np.zeros((15, 15))
                features.append(b)

        x1 = np.asarray(features)
        x1 = torch.tensor(x1, dtype=torch.float32)
        x1 = x1.view(1, x1.shape[0], x1.shape[1], x1.shape[2])
        turn_out = torch.tensor([0,1], dtype=torch.float32)
        turn_out = turn_out.view(1,turn_out.shape[0])
        action = self.model_actor(x1, turn_out)
        mask = copy.deepcopy(boardEmpty)

        # print('chess list in chess', self.chessList)
        # print('zero index', np.where(mask == 0))
        zeroindex = np.where(mask == 0)
        # for i in range(zeroindex[0].size):
        #     print('zero index', zeroindex[0][i],  zeroindex[1][i])
        mask = mask.reshape(1,15*15)

        action_p = action

        m = nn.Softmax(dim=1)
        action_p = m(action_p)
        action_p = action_p * torch.tensor(mask,dtype=torch.float32)
        action_display = action_p
        action_display = action_display.reshape(15, 15)
        azeroindex = np.where(action_display == 0)
        # for i in range(azeroindex[0].size):
        #     print('action zero index', azeroindex[0][i], azeroindex[1][i])
        action_index = action_p.multinomial(num_samples=1, replacement=True)
        f_action_index = action_index.detach().numpy()[0][0]
        y = f_action_index // 15
        x = f_action_index % 15

        # print('action in chess', y,x)
        # return [x,y]

        return [y,x]



def main():
    a = [3,4,5,1,2]
    mask = [1,1,0,0,0]
    a_numpy = np.asarray(a)
    a_tensor = torch.tensor(a_numpy, dtype=torch.float32).view(1,5)
    mask_numpy = np.asarray(mask)
    mask_tensor = torch.tensor(mask_numpy, dtype=torch.float32).view(1,5)
    p_tensor = a_tensor*mask_tensor
    print(p_tensor)
    m = nn.Softmax(dim=1)
    action_p = m(p_tensor)
    action_index = action_p.multinomial(num_samples=1, replacement=True)

    print(action_p)
    print(action_index)

if __name__ == '__main__':
    main()

