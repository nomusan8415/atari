import gym
import numpy as np
import gym.spaces
import pySaliencyMapDefs
import pySaliencyMap
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
import chainer
import chainerrl
import chainer.functions as F
import chainer.links as L
import cupy as cp
import gym.spaces
from scipy import misc

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

ENV_NAME = 'MsPacman-v0'
env = gym.make(ENV_NAME)
obs = env.reset()
path = "/home/nomura/Research/atari/image-saliency-3/"

env.frameskip = 4
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(n_actions)
#q_func.to_gpu(0)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 5*10**4)
phi = lambda x:x.astype(dtype=cp.float32)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, phi=phi
    )
obs = obs[:,:,0]
obs = (misc.imresize(obs, (110, 84)))[110-84-8:110-8,:]
obs4steps = np.zeros((4,84,84), dtype=np.float32)
obs4steps[0] = obs
obs4steps[1] = obs
obs4steps[2] = obs
obs4steps[3] = obs
reward=0

for i in range(300):
    action = agent.act_and_train(obs4steps, reward)
    obs, reward, done, _ = env.step(action)
    img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
#env.env.ale.saveScreenPNG('saliency.jpg')
    img = img[0 : 171, 0 : 160]
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    height, width, c = img.shape

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")


#print saliencyMap
    #max_loc = []
    #for y in range(height):
    #    for x in range(width):
    #        if saliencyMap[y,x] >= 0.9:
    #            max_loc.append([x,y])
    #            #print max_loc

    #cut_num = len(max_loc)
    #print cut_num
    #for n in range(cut_num):
    #    img = cv2.circle(img,tuple(max_loc[n]),20,(0,0,255),1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saliencyMap)
    img = cv2.circle(img,tuple(max_loc),20,(0,0,255),1)
    cv2.imwrite(path + str(i) + ".png", img)
