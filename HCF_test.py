import gym
import os
import glob
import numpy as np
from datetime import datetime
import codecs
import json
import cv2
from matplotlib import pyplot as plt
from HCF_functions import get_paddle_position, get_max_tunnel_depth, get_ball_position


# This wrapper extracts Hand Crafted Features from gym observations
class HCFgymWrapper(gym.ObservationWrapper):
    # __init__ save the given inputs and creates the needed directories to save results
    def __init__(self, env, FuncList, DstDir=os.getcwd()):
        super().__init__(env)
        self.FuncList = FuncList
        self.DstDir = DstDir
        self.resultsDir = os.path.join(self.DstDir, "results")
        self.Outputs = dict()
        if not os.path.exists(self.DstDir):
            os.mkdir(self.DstDir)
            os.mkdir(os.path.join(self.resultsDir))
        elif not os.path.exists(os.path.join(self.resultsDir)):
            os.mkdir(os.path.join(self.resultsDir))
        for func in self.FuncList:
            self.Outputs[func.__name__] = []

    # This function overrides the observation function of gym.
    # First, the function extracts hand-crafted features, by activating all the
    #       functions in self.FuncList on the observation.
    # Then, it returns the observation as was seen by the agent.
    def observation(self, obs):
        # modify obs
        for func in self.FuncList:
            res = func(obs)
            if isinstance(res, np.ndarray):
                self.Outputs[func.__name__].append(res.tolist())
            else:
                self.Outputs[func.__name__].append(res)
        return obs

    # close function is called when the gym environment is closed and saves the results
    def close(self):
        now = str(datetime.now().strftime("%m.%d.%Y_%H:%M:%S"))
        results_file = os.path.join(self.resultsDir, "HandCraftedFeatures"+now+".json")
        json.dump(self.Outputs, codecs.open(results_file, 'w', encoding='utf-8'))
        return super().close()

FuncList = []
# FuncList.append(get_paddle_position)
FuncList.append(get_max_tunnel_depth)

env = gym.make("Breakout-v4")
Wrap = HCFgymWrapper(env, FuncList=FuncList)

obs_list = []
T = 500
s_t = Wrap.reset()
for t in range(T):
    a_t = Wrap.action_space.sample()
    observations, rewards, dones, infos = Wrap.step(a_t)
    obs_list.append(observations)
    if dones:
        s_t = Wrap.reset()

# rnd = np.round(np.random.randint(T, size=1))[0]
# im = obs_list[rnd].copy()
# cY, cX = get_ball_position(im)
# im[int(cY)-2:int(cY)+2, int(cX)-2:int(cX)+2, :] = 150
# plt.imshow(im)
# plt.show()

# *************** Uncomment to show results ***************
rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols)
plt.subplots_adjust(hspace=0.4)
fig.suptitle('observation and tunnel depth \n paddle in white and ball in gray')

for i in range(rows*cols):
    rnd = np.round(np.random.randint(T, size=1))[0]
    im = obs_list[rnd]
    max_depth, tunnel_open, all_depths = get_max_tunnel_depth(im)
    cY, cX = get_paddle_position(im)
    ballY, ballX = get_ball_position(im)
    if all([ballY != 0, ballX != 0]):
        im[int(ballY) - 2:int(ballY) + 2, int(ballX) - 2:int(ballX) + 2, :] = 150
    im[int(cY)-2:int(cY)+2, int(cX)-2:int(cX)+2, :] = 255
    axs.ravel()[i].imshow(im)
    axs.ravel()[i].set_title("#obs: " + str(rnd) + " max_depth: " + str(max_depth) +
                             (" tunnel is open" if tunnel_open else " tunnel is closed"))
plt.show()


# *************** Uncomment to close and save results ***************

# Wrap.close()
#
# resultsDir = Wrap.resultsDir
# list_of_files = glob.glob(str(resultsDir+"/*"))     # * means all if need specific format then *.csv
# latest_file = max(list_of_files, key=os.path.getctime)
# print("will open this file:"+latest_file)
#
# resultsFile = os.path.abspath(latest_file)
# assert os.path.exists(resultsFile)
# with open(resultsFile, "r") as read_file:
#     data = json.load(read_file)
# data_ndarr = np.array(data['get_padel_position'])
# print(data_ndarr.shape)
#
# rows, cols = np.where(data_ndarr != [7.5, 76.5])
# print(data_ndarr[rows])
