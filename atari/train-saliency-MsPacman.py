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
import sys
import traceback

path = "result/saliency-MsPacman-4.txt"
path2 = "result/saliency-MsPacman-4-testplay.txt"
path3 = "image-original/"
path4 = "image-saliency/"
h, w = 30, 30

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
	#print self.L4(h)
        return chainerrl.action_value.DiscreteActionValue(self.L4(h))

def blur(x1, y1, x2, y2, r, f, img):
    if f == 1.0 :
        blurred_img = img
    else :
        if f % 2 == 0 :
            f = f + 1
        blurred_img = cv2.GaussianBlur(img, (f,f), 0)
    if x1 == 0 and y1 == 0 and x2 == img.shape[1] and y2 == img.shape[0]:
        return blurred_img
    crop_img = blurred_img[y1 : y2 , x1 : x2]

    next_y1 = 0 if y1-r < 0 else y1-r
    next_y2 = y2+r if y2+r <= img.shape[0] else img.shape[0]
    next_x1 = 0 if x1-r < 0 else x1-r
    next_x2 = x2+r if x2+r <= img.shape[1] else img.shape[1]
    f+=7
    res_img = blur(next_x1, next_y1, next_x2, next_y2, r, f, img)
    res_img[y1 : y2 , x1 : x2] = crop_img
    return res_img

def saliency(img):
    height, width, c = img.shape
    #顕著性検出
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = sal.computeSaliency(img)
    #最大値検出
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saliencyMap)
    return max_loc

def test_play():
    f2 = open(path2, 'a')
    total_reward = 0
    for j in range(3):
        obs4steps = np.zeros((4,84,84), dtype=np.float32)
        obs = env.reset()
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        obs = obs[0 : 170, 0 : 160]
        obs = cv2.resize(obs, (obs.shape[1]*2, obs.shape[0]*2))
        #顕著性検出,最も顕著性が高い座標を返す
        max_loc = saliency(obs)
        y1 = 0 if max_loc[1]-h < 0 else max_loc[1]-h
        y2 = max_loc[1]+h if max_loc[1]+h < obs.shape[0] else obs.shape[0]-1
        x1 = 0 if max_loc[0]-w < 0 else max_loc[0]-w
        x2 = max_loc[0]+w if max_loc[0]+w < obs.shape[1] else obs.shape[1]-1
	#print y1,y2,x1,x2
        #切り出し
        center = obs[y1 : y2, x1 : x2]
        #グラデーションぼかし
        obs = blur(x1, y1, x2, y2, 50, 0, obs)
	#print obs.shape
	#print center.shape
        #貼り付け
        obs[y1 : y2, x1 : x2] = center
        obs = obs[:,:,0]
        obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]
        obs4steps[0] = obs
        r = 0
        done = False
        R = 0
        t = 0
        while not done:
            env.render()
            action = agent.act(obs4steps)
            obs, r, done, _ = env.step(action)
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            obs = obs[0 : 170, 0 : 160]
            obs = cv2.resize(obs, (obs.shape[1]*2, obs.shape[0]*2))
            #顕著性検出,最も顕著性が高い座標を返す
            max_loc = saliency(obs)
            y1 = 0 if max_loc[1]-h < 0 else max_loc[1]-h
            y2 = max_loc[1]+h if max_loc[1]+h < obs.shape[0] else obs.shape[0]-1
            x1 = 0 if max_loc[0]-w < 0 else max_loc[0]-w
            x2 = max_loc[0]+w if max_loc[0]+w < obs.shape[1] else obs.shape[1]-1
            #切り出し
            center = obs[y1 : y2, x1 : x2]
            #グラデーションぼかし
            obs = blur(x1, y1, x2, y2, 50, 0, obs)
            #貼り付け
            obs[y1 : y2, x1 : x2] = center
            obs = obs[:,:,0]
            obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]
            obs4steps = np.roll(obs4steps, 1, axis=0)
            obs4steps[0] = obs
            R += r
            t += 1
        print('test play:', j, 'R:', R)
        total_reward += R
    f2.write(str(total_reward/3) + "\n")
    f2.close()




ENV_NAME = 'MsPacman-v0'
NUM_EPISODES = 13000

env = gym.make(ENV_NAME)#環境指定
env.frameskip = 4
#q関数初期化,パラメータ設定\
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(n_actions)
q_func.to_gpu(0)
optimizer = chainer.optimizers.RMSprop(lr=0.00025,alpha=0.95, eps=0.01)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
gamma = 0.99
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.1, decay_steps=1200000, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**5)
phi = lambda x:x.astype(dtype=cp.float32)##型の変換(chainerはfloat32型。float64は駄目)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, phi=phi
    )
#agent.load('saliency-MsPacman-2')

start = time.time()

j=0

try :
    for i in range(NUM_EPISODES):
        #if(i > 0 and i % 50 == 0):
        #    test_play()
        f = open(path, 'a')
        obs4steps = np.zeros((4,84,84), dtype=np.float32)
        obs = env.reset()
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        obs = obs[0 : 170, 0 : 160]
        obs = cv2.resize(obs, (obs.shape[1]*2, obs.shape[0]*2))
        #顕著性検出,最も顕著性が高い座標を返す
        max_loc = saliency(obs)
        y1 = 0 if max_loc[1]-h < 0 else max_loc[1]-h
        y2 = max_loc[1]+h if max_loc[1]+h < obs.shape[0] else obs.shape[0]-1
        x1 = 0 if max_loc[0]-w < 0 else max_loc[0]-w
        x2 = max_loc[0]+w if max_loc[0]+w < obs.shape[1] else obs.shape[1]-1
        #切り出し
        center = obs[y1 : y2, x1 : x2]
        #グラデーションぼかし
        obs = blur(x1, y1, x2, y2, 50, 0, obs)
        #貼り付け
        obs[y1 : y2, x1 : x2] = center
        obs = obs[:,:,0]
        obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]

        obs4steps[0] = obs
        reward = 0
        total_reward = 0
        done = False
        R = 0
        t = 0
        frame = 0
        while not done:
            j += 1
            env.render()
            action = agent.act_and_train(obs4steps, reward)
            obs, reward, done, _ = env.step(action)
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            obs = obs[0 : 170, 0 : 160]
            obs = cv2.resize(obs, (obs.shape[1]*2, obs.shape[0]*2))
            cv2.imwrite(path3 + str(j) + ".png", obs)
            #顕著性検出,最も顕著性が高い座標を返す
            max_loc = saliency(obs)
            y1 = 0 if max_loc[1]-h < 0 else max_loc[1]-h
            y2 = max_loc[1]+h if max_loc[1]+h < obs.shape[0] else obs.shape[0]-1
            x1 = 0 if max_loc[0]-w < 0 else max_loc[0]-w
            x2 = max_loc[0]+w if max_loc[0]+w < obs.shape[1] else obs.shape[1]-1
            #切り出し
            center = obs[y1 : y2, x1 : x2]
            #グラデーションぼかし
            obs = blur(x1, y1, x2, y2, 30, 0, obs)
            #貼り付け
            obs[y1 : y2, x1 : x2] = center
            cv2.imwrite(path4 + str(j) + ".png", obs)
            obs = obs[:,:,0]
            obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]
            obs4steps = np.roll(obs4steps, 1, axis=0)
            obs4steps[0] = obs
            R += reward
            t += 1
            frame += 1
        print('episode:', i,'R:', R)
        print('time:', time.time() - start)
        print('frame:', frame)
        agent.stop_episode_and_train(obs4steps, reward, done)
        f.write(str(R) + "\n")
        f.close()
        #if(i % 50 == 0):
        #    agent.save('saliency-MsPacman-4')

    print('Finished')
    #agent.save('saliency-MsPacman-4')
except Exception as e:
    traceback.print_exc()
    print('Interrupted')
    f.close()
    #agent.save('saliency-MsPacman-4')
