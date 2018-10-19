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

def saliency(img):
    height, width, c = img.shape
    #顕著性検出
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = sal.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    #最大値検出
    max_loc = []
    max_loc = [[x,y] for y in range(1,height-1) for x in range(1,width-1) if (saliencyMap[y,x] >= 230 and saliencyMap[y,x+1] >= 230 and saliencyMap[y,x-1] >= 230 and saliencyMap[y+1,x] >= 230 and saliencyMap[y-1,x] >= 230)]
    return max_loc


path = "result/saliency-MsPacman-dqn.txt"
#path2 = "/home/nomura/Research/atari/image/"

ENV_NAME = 'MsPacman-v0'
NUM_EPISODES = 10000

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
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**5)
phi = lambda x:x.astype(dtype=cp.float32)##型の変換(chainerはfloat32型。float64は駄目)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, phi=phi
    )
#agent.load('MsPacman-dqn')

start = time.time()
h, w = 30, 30
j=0
cut_loc = []
f = 15

try :
    for i in range(NUM_EPISODES):
        f = open(path, 'a')
        obs4steps = np.zeros((4,84,84), dtype=np.float32)
        obs = env.reset()
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        obs = obs[0 : 170, 0 : 160]
        obs = cv2.resize(obs, (obs.shape[1]*2, obs.shape[0]*2))
        #顕著性検出,最も顕著性が高い座標を返す
        top_loc = saliency(obs)
        cut_num = len(top_loc)
        #切り出し
        #for n in range(cut_num) :
        #    #print top_loc[n][1]-h,  top_loc[n][1]+h, top_loc[n][0]-w, top_loc[n][0]+w
        #    cut_loc.append(obs[top_loc[n][1]-h: top_loc[n][1]+h, top_loc[n][0]-w : top_loc[n][0]+w])
        cut_loc = []
        cut_loc = [obs[top_loc[n][1]-h: top_loc[n][1]+h, top_loc[n][0]-w : top_loc[n][0]+w] for n in range(cut_num)]
        #ぼかし
        obs = cv2.GaussianBlur(obs, (35,35), 0)
        #obs = blur(top_loc[0]-w, top_loc[1]-h, top_loc[0]+w, top_loc[1]+h, 30, 10)
        #自機画像貼り付け
        for n in range(cut_num) :
            obs[top_loc[n][1]-h: top_loc[n][1]+h, top_loc[n][0]-w : top_loc[n][0]+w] = cut_loc[n]
        obs = obs[:,:,0]
        obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]

        obs4steps[0] = obs
        reward = 0
        total_reward = 0
        done = False
        R = 0
        t = 0
        while not done:
            j += 1
            env.render()
            action = agent.act_and_train(obs4steps, reward)
            obs, reward, done, _ = env.step(action)
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            obs = obs[0 : 170, 0 : 160]
            obs = cv2.resize(obs, (obs.shape[1]*2, obs.shape[0]*2))
            #顕著性検出,最も顕著性が高い座標を返す
            top_loc = saliency(obs)
            cut_num = len(top_loc)
            cut_loc = []
            #切り出し
            #for n in range(cut_num) :
            #    #print top_loc[n][1]-h,  top_loc[n][1]+h, top_loc[n][0]-w, top_loc[n][0]+w
            #    cut_loc.append(obs[top_loc[n][1]-h: top_loc[n][1]+h, top_loc[n][0]-w : top_loc[n][0]+w])
            cut_loc = [obs[top_loc[n][1]-h: top_loc[n][1]+h, top_loc[n][0]-w : top_loc[n][0]+w] for n in range(cut_num)]
            #ぼかし
            obs = cv2.GaussianBlur(obs, (35,35), 0)
            #obs = blur(top_loc[0]-w, top_loc[1]-h, top_loc[0]+w, top_loc[1]+h, 30, 10)
            #自機画像貼り付け
            for n in range(cut_num) :
                obs[top_loc[n][1]-h: top_loc[n][1]+h, top_loc[n][0]-w : top_loc[n][0]+w] = cut_loc[n]
            #obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(path2 + str(j) + ".png", obs)
            obs = obs[:,:,0]
            obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]
            obs4steps = np.roll(obs4steps, 1, axis=0)
            obs4steps[0] = obs
            R += reward
            t += 1
        print('episode:', i,'R:', R)
        print('time:', time.time() - start)
        agent.stop_episode_and_train(obs4steps, reward, done)
        f.write(str(R) + "\n")
        f.close()

    print('Finished')

    agent.save('saliency-MsPacman-dqn')

except Exception as e:
    print('Interrupted')
    print >> sys.stderr, 'Error occurred!'
    print '=== エラー内容 ==='
    print 'type:' + str(type(e))
    print 'args:' + str(e.args)
    print 'message:' + e.message
    print 'e自身:' + str(e)
    f.close()
    agent.save('saliency-MsPacman-dqn')
