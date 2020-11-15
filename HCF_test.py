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
        self.acc_reward = 0
        if not os.path.exists(self.DstDir):
            os.mkdir(self.DstDir)
            os.mkdir(os.path.join(self.resultsDir))
        elif not os.path.exists(os.path.join(self.resultsDir)):
            os.mkdir(os.path.join(self.resultsDir))
        for func in self.FuncList:
            self.Outputs[func.__name__] = []
        self.Outputs['Score'] = []
        self.Outputs['Lives'] = []

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

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.acc_reward += reward
        if isinstance(self.acc_reward, np.ndarray):
            self.Outputs['Score'].append(self.acc_reward.tolist())
        else:
            self.Outputs['Score'].append(self.acc_reward)

        if isinstance(info['ale.lives'], np.ndarray):
            self.Outputs['Lives'].append(info['ale.lives'].tolist())
        else:
            self.Outputs['Lives'].append(info['ale.lives'])
        return next_state, reward, done, info

    def reset(self):
        self.acc_reward = 0
        return super().reset()

    # close function is called when the gym environment is closed and saves the results
    def close(self):
        now = str(datetime.now().strftime("%m.%d.%Y_%H:%M:%S"))
        results_file = os.path.join(self.resultsDir, "HandCraftedFeatures"+now+".json")
        json.dump(self.Outputs, codecs.open(results_file, 'w', encoding='utf-8'))
        return super().close()


FuncList = [get_paddle_position, get_max_tunnel_depth, get_ball_position]

env = gym.make("Breakout-v4")
Wrap = HCFgymWrapper(env, FuncList=FuncList)

obs_list = []
T = 5000
s_t = Wrap.reset()
for t in range(T):
    a_t = Wrap.action_space.sample()
    observations, rewards, dones, infos = Wrap.step(a_t)
    obs_list.append(observations)
    if dones:
        s_t = Wrap.reset()

# rnd = np.round(np.random.randint(T, size=1))[0]
# im = obs_list[rnd].copy()
# area_Y_start = 0
# area_Y_end = 17
# numbers_area = im[area_Y_start:area_Y_end, :]
# plt.imshow(obs_list[0])
# plt.show()

# *************** Uncomment to show results ***************
rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols)
plt.subplots_adjust(hspace=0.4)
fig.suptitle('Hand crafted features \n paddle in white and ball in gray')

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
    axs.ravel()[i].set_title("#obs: " + str(rnd) + ", max_depth: " + str(max_depth) +
                             (", tunnel is open" if tunnel_open else ", tunnel is closed") +
                             "\n" + "Score is: " + str(Wrap.Outputs['Score'][rnd]) + "\n Lives: " + str(Wrap.Outputs['Lives'][rnd]))
plt.show()


# *************** Uncomment to close and save results ***************

Wrap.close()  # Write the results file

# Now we'll open the saved file to make sure it looks good
resultsDir = Wrap.resultsDir
list_of_files = glob.glob(str(resultsDir+"/*"))     # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print("will open this file:"+latest_file)

resultsFile = os.path.abspath(latest_file)
assert os.path.exists(resultsFile)
with open(resultsFile, "r") as read_file:
    data = json.load(read_file)

# Just an example
data_ndarr = np.array(data['get_paddle_position'])
print(data_ndarr.shape)
