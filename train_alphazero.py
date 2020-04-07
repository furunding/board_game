import gym
import sys
import logging
from ai.alphazero import AlphaZeroAgent, self_play
from ai import EvaluateAI
from test_env import GomukuEnv

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s')

def test(agent):
    win = 0
    for k in range(20):
        env = GomukuEnv(board_size=6, target_length=4, opponent=EvaluateAI)
        observation = env.reset()

        for i in range(1000):
            action = agent.decide(observation)
            observation, reward, done, info = env.step(action)
            if done:
                if reward == 1:
                    win += 1
                break
    print("win rate: {}".format(win / 20.0))
    env.close()

if __name__ == "__main__":
    """
    AlphaZero 参数，可用来求解比较大型的问题（如五子棋）
    """
    # train_iterations = 700000 # 训练迭代次数
    # train_episodes_per_iteration = 5000 # 每次迭代自我对弈回合数
    # batches = 10 # 每回合进行几次批学习
    # batch_size = 4096 # 批学习的批大小
    # sim_count = 800 # MCTS需要的计数
    # net_kwargs = {}
    # net_kwargs['conv_filters'] = [256,]
    # net_kwargs['residual_filters'] = [[256, 256],] * 19
    # net_kwargs['policy_filters'] = [256,]

    """
    小规模参数，用来初步求解比较小的问题（如井字棋）
    """
    train_iterations = 100
    # train_episodes_per_iteration = 100
    train_episodes_per_iteration = 50
    batches = 8
    batch_size = 64
    sim_count = 200
    # sim_count = 200
    net_kwargs = {}
    # net_kwargs['conv_filters'] = [256,]
    # net_kwargs['residual_filters'] = [[256, 256],]
    # net_kwargs['policy_filters'] = [256,]

    net_kwargs['conv_filters'] = [64,]
    net_kwargs['residual_filters'] = [[64, 64], [64, 64]]
    net_kwargs['policy_filters'] = [64,]

    env = gym.make('Gomuku-v0', board_shape=(6, 6), target_length=4)

    agent = AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=sim_count,
            batches=batches, batch_size=batch_size)

    for iteration in range(train_iterations):
        # 自我对弈
        dfs_trajectory = []
        for episode in range(train_episodes_per_iteration):
            df_trajectory = self_play(env, agent,
                    return_trajectory=True, verbose=False)
            logging.info('训练 {} 回合 {}: 收集到 {} 条经验'.format(
                    iteration, episode, len(df_trajectory)))
            dfs_trajectory.append(df_trajectory)

        # 利用经验进行学习
        agent.learn(dfs_trajectory)
        logging.info('训练 {}: 学习完成'.format(iteration))

        # # 演示训练结果
        # self_play(env, agent, verbose=True)

        # 测试胜率
        test(agent)