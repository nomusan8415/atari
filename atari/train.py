# -*- coding: utf-8 -*-
import chainer
import chainerrl
import chainer.functions as F
import chainer.links as L
import gym
import time
import numpy as np
import cupy as cp
from datetime import datetime
import gym.spaces
from scipy import misc
import cv2

class QFunction(chainer.Chain):
    def __init__(self, n_actions):
        super(QFunction, self).__init__(
            L0=L.Convolution2D(4 , 32, ksize=8, stride=4),
            L1=L.Convolution2D(32, 64, ksize=4, stride=2),
            L2=L.Convolution2D(64, 64, ksize=3, stride=1),
            L3=L.Linear(3136, 512),
            L4=L.Linear(512, n_actions))

    def __call__(self, x, test=False):
        h = F.relu(self.L0(x))
        h = F.relu(self.L1(h))
        h = F.relu(self.L2(h))
        h = F.relu(self.L3(h))
        return chainerrl.action_value.DiscreteActionValue(self.L4(h))

def blur(x1, y1, x2, y2, r, f):
    if f == 1.0 :
        blurred_img = obs
    else :
        if f % 2 == 0 :
            f = f + 1
        blurred_img = cv2.GaussianBlur(obs, (f,f), 0)
    if x1 == 0 and y1 == 0 and x2 == obs.shape[1] and y2 == obs.shape[0]:
        return blurred_img
    crop_img = blurred_img[y1 : y2 , x1 : x2]

    next_y1 = 0 if y1-r < 0 else y1-r
    next_y2 = y2+r if y2+r <= obs.shape[0] else obs.shape[0]
    next_x1 = 0 if x1-r < 0 else x1-r
    next_x2 = x2+r if x2+r <= obs.shape[1] else obs.shape[1]
    f+=7
    res_img = blur(next_x1, next_y1, next_x2, next_y2, r, f)
    res_img[y1 : y2 , x1 : x2] = crop_img
    return res_img

template = cv2.imread("voxing_template.png")
h, w,  = template.shape[:2]
path = 'result/boxing-dqn.txt"


ENV_NAME = 'MsPacman-v0'
NUM_EPISODES = 3000

env = gym.make(ENV_NAME)#環境指定
env.frameskip = 4
#q関数初期化,パラメータ設定\
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(n_actions)
q_func.to_gpu(0)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 5*10**4)
phi = lambda x:x.astype(dtype=cp.float32)##型の変換(chainerはfloat32型。float64は駄目)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, phi=phi
    )

#agent.load('Boxing-v0')

start = time.time()
try :
    for i in range(NUM_EPISODES):
        f = open(path, 'a')
        obs4steps = np.zeros((4,84,84), dtype=np.float32)
        obs = env.reset()
        #自機検出
        res = cv2.matchTemplate(obs, template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        #切り出し
        center = obs[top_left[1]: top_left[1]+h, top_left[0] : top_left[0]+w]
        #グラデーションぼかし
        obs = blur(top_left[0], top_left[1], top_left[0]+w, top_left[1]+h, 30, 0)
        #自機画像貼り付け
        obs[top_left[1]: top_left[1]+h, top_left[0] : top_left[0]+w] = center
        obs = obs[:,:,0]
        obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]

        obs4steps[0] = obs
        reward = 0
        total_reward = 0
        done = False
        R = 0
        t = 0
        while not done:
            env.render()
            action = agent.act_and_train(obs4steps, reward)
            obs, reward, done, _ = env.step(action)
            res = cv2.matchTemplate(obs, template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            #切り出し
            center = obs[top_left[1]: top_left[1]+h, top_left[0] : top_left[0]+w]
            #グラデーションぼかし
            obs = blur(top_left[0], top_left[1], top_left[0]+w, top_left[1]+h, 30, 0)
            #自機画像貼り付け
            obs[top_left[1]: top_left[1]+h, top_left[0] : top_left[0]+w] = center
            obs = obs[:,:,0]
            obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]
            obs4steps = np.roll(obs4steps, 1, axis=0)
            obs4steps[0] = obs
            R += reward
            t += 1
        print('episode:', i,
            'R:', R)
        print('time:', time.time() - start)
        agent.stop_episode_and_train(obs4steps, reward, done)
        f.write(str(R) + "\n")
        f.close()

    print('Finished')

    agent.save('Boxing-dqn')

except Exception as e:
    print('Interrupted')
    f.close()
    agent.save('boxing-dqn')
