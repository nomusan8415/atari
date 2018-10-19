# -*- coding: utf-8 -*-
import sys

import gym
import time



ENV_NAME = 'shooting-v0'
NUM_EPISODES = 12000
NO_OP_STEPS = 30

env = gym.make(ENV_NAME)#環境指定
start = time.time()

#q関数初期化,パラメータ設定\
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**6)
phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_frequency=1,
    target_update_frequency=100, phi=phi)


observation = env.start(NUM_EPISODES)
#学習スタート
for _ in range(NUM_EPISODES):
    done = False #エピソード終了判定を初期化
    reward = o #報酬初期化

    while not done:
        last_observation = observation
        action = agent.act_and_train(observation, reward)#行動選択
        observation, reward, done = env.step(action)#実行,次の画像と報酬が帰ってくる

    agent.stop_episode_and_train(observation, reward, done)
    obs = env.reset()#環境初期化

agent.save('model')
print "train finished"
