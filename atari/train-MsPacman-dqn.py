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

path = "result/MsPacman-dqn-2.txt"

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
try :
    for i in range(NUM_EPISODES):
        f = open(path, 'a')
        obs4steps = np.zeros((4,84,84), dtype=np.float32)
        obs = env.reset()
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
            img = cv2.imread("pacman-image/20.png")
            img = img[0 : 171, 0 : 160]
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
            height, width, c = img.shape
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMap) = saliency.computeSaliency(img)
            saliencyMap = (saliencyMap * 255).astype("uint8")
            #threshMap = cv2.threshold(saliencyMap, 0, 255,
            #	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            max_loc = []
            #for y in range(height):
            #    for x in range(width):
            #        print "a"
            max_loc = [[y,x] for y in range(height) for x in range(width) if (saliencyMap[y,x] >= 230 and saliencyMap[y,x+1] >= 230 and saliencyMap[y,x-1] >= 230 and saliencyMap[y+1,x] >= 230 and saliencyMap[y-1,x] >= 230)]
                    #if saliencyMap[y,x] >= 230 and saliencyMap[y,x+1] >= 230 and saliencyMap[y,x-1] >= 230 and saliencyMap[y+1,x] >= 230 and saliencyMap[y-1,x] >= 230:
                        #max_loc.append([x,y])
            cut_num = len(max_loc)
            print len(max_loc)
            #for n in range(cut_num) :
            #    img = cv2.circle(img,tuple(max_loc[n]),20,(0,0,255),1)
            img = [cv2.circle(img,tuple(max_loc[n]),20,(0,0,255),1) for n in range(cut_num)]
            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saliencyMap)
            #img = cv2.circle(img,(max_loc[0],max_loc[1]),20,(0,0,255),1)

            action = agent.act_and_train(obs4steps, reward)
            obs, reward, done, _ = env.step(action)
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

    agent.save('MsPacman-dqn-2')

except Exception as e:
    print('Interrupted')
    print '=== エラー内容 ==='
    print 'type:' + str(type(e))
    print 'args:' + str(e.args)
    print 'message:' + e.message
    print 'e自身:' + str(e)
    f.close()
    agent.save('MsPacman-dqn-2')
