import gym
import os
import glob
import numpy as np
import json
from matplotlib import pyplot as plt
from HCF_functions import get_paddle_position, get_max_tunnel_depth, get_ball_position
from HCF_gym_wrapper import HCFgymWrapper


FuncList = [get_paddle_position, get_max_tunnel_depth, get_ball_position]

env = gym.make("Breakout-v4")
Wrap = HCFgymWrapper(env, FuncList=FuncList)

# use random actions just to gather observations
obs_list = []
T = 5000
s_t = Wrap.reset()
for t in range(T):
    a_t = Wrap.action_space.sample()
    observations, rewards, dones, infos = Wrap.step(a_t)
    obs_list.append(observations)
    if dones:
        s_t = Wrap.reset()

# # show a single image and test
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
