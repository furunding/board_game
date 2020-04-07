from test_env import PlayGomuku
from ai import MCTSPlayer, MCTSParallelPlayer, RandomChoose, EvaluateAI
import gym
from boardgame2 import strfboard

if __name__ == "__main__":
    board_size = 8
    target_length = 5
    test = PlayGomuku(
        board_size=board_size,
        target_length=target_length,
        black_agent=MCTSParallelPlayer,
        white_agent=RandomChoose,
    )

    test.run(episodes=20, verbose=True)