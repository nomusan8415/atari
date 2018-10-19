import chainer
import chainerrl
import gym
import time
import numpy as np
import cupy as cp
from datetime import datetime
import gym.spaces
from scipy import misc
import cv2



def blur(x1, y1, x2, y2, r, f):
    if f == 1.0 :
        blurred_img = img
    else :
        if f % 2 == 0 :
            f = f + 1
        blurred_img = cv2.GaussianBlur(img, (f,f), 0)
    print x1,y1,x2,y2
    if x1 == 0 and y1 == 0 and x2 == img.shape[1] and y2 == img.shape[0]:
        return blurred_img
    crop_img = blurred_img[y1 : y2 , x1 : x2]

    next_y1 = 0 if y1-r < 0 else y1-r
    next_y2 = y2+r if y2+r <= img.shape[0] else img.shape[0]
    next_x1 = 0 if x1-r < 0 else x1-r
    next_x2 = x2+r if x2+r <= img.shape[1] else img.shape[1]
    f+=5
    res_img = blur(next_x1, next_y1, next_x2, next_y2, r, f)
    res_img[y1 : y2 , x1 : x2] = crop_img
    return res_img


#ENV_NAME = 'MsPacman-v0'
template=[[0 for i in range(3)]for j in range(5)]
template[0] = cv2.imread("pacman-template.png")
#print template.shape
#template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
template[1] = cv2.imread("pacman-template2.png")
#print template2.shape
template[2] = cv2.imread("pacman-template3.png")
#print template3.shape
template[3] = cv2.imread("pacman-template4.png")
#print template3.shape
val = 0
h=0
w=0

#env = gym.make(ENV_NAME)
#img = env.reset()
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#cv2.imwrite("pacman.png", img)
#ew, eh, c = img.shape[:3]
#print img.shape
img = cv2.imread("image/2018-07-28 18:28:58.395887.png")
img2 = img[0 : 171, 0 : 160]
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for i in range(3) :
    if i == 0 :
        res = cv2.matchTemplate(img2, template[0],cv2.TM_CCOEFF_NORMED)

    elif i == 1 :
        res = cv2.matchTemplate(img2, template[1],cv2.TM_CCOEFF_NORMED)
    elif i == 2 :
        res = cv2.matchTemplate(img2, template[2],cv2.TM_CCOEFF_NORMED)
    else :
        res = cv2.matchTemplate(img2, template[3],cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if val < max_val :
        val = max_val
        loc = max_loc
        h,w = h, w,  = template[i].shape[:2]
print val
top_left = loc
center = img[top_left[1]: top_left[1]+h, top_left[0] : top_left[0]+w]

img = blur(top_left[0], top_left[1], top_left[0]+w, top_left[1]+h, 30, 0)
img[top_left[1]: top_left[1]+h, top_left[0] : top_left[0]+w] = center


cv2.imwrite("mat.png", img)

#env.env.ale.saveScreenPNG(b'voxing_image.png')
