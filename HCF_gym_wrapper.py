import gym
import os
import codecs
import json
from datetime import datetime
import HCF_functions
DFLT_FUNC_LIST = [HCF_functions.get_ball_position, HCF_functions.get_paddle_position,
                  HCF_functions.get_max_tunnel_depth]
DFLT_LIVES_COUNT = 5


# This wrapper extracts Hand Crafted Features from gym observations
class HCFgymWrapper(gym.ObservationWrapper):
    # __init__ save the given inputs and creates the needed directories to save results
    def __init__(self, env, FuncList=DFLT_FUNC_LIST, DstDir=os.getcwd()):
        super().__init__(env)
        self.FuncList = FuncList
        self.DstDir = DstDir
        self.resultsDir = os.path.join(self.DstDir, "results")
        self.Outputs = dict()
        self.acc_reward = 0  # Accumulated reward in a session. When the last life is lost, resetted to zero.
        if not os.path.exists(self.DstDir):  # Create missing folders
            os.mkdir(self.DstDir)
            os.mkdir(os.path.join(self.resultsDir))
        elif not os.path.exists(os.path.join(self.resultsDir)):
            os.mkdir(os.path.join(self.resultsDir))
        # Initiate lists of results for json compatibility
        for func in self.FuncList:
            self.Outputs[func.__name__] = []
        self.Outputs['Score'] = []
        self.Outputs['Lives'] = []

    # This function overrides the observation function of gym.
    # First, the function extracts hand-crafted features, by activating all the
    #       functions in self.FuncList on the observation.
    # Then, it returns the observation as was seen by the agent.
    def observation(self, obs):
        for func in self.FuncList:
            res = func(obs)
            self.Outputs[func.__name__].append(res)
        return obs

    # This function overrides the step function of gym, to allow extraction of the score and lives count.
    # These are used as Hand Crafted Features.
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.observation(next_state)
        self.acc_reward += reward
        self.Outputs['Score'].append(self.acc_reward)
        self.Outputs['Lives'].append(info['ale.lives'])
        return next_state, reward, done, info

    # This function overrides the reset function of gym, to allow resetting of the score, and updating the features.
    def reset(self):
        self.acc_reward = 0
        self.Outputs['Score'].append(self.acc_reward)
        self.Outputs['Lives'].append(DFLT_LIVES_COUNT)
        return super().reset()

    # close function is called when the gym environment is closed and saves the results
    def close(self):
        now = str(datetime.now().strftime("%m.%d.%Y_%H:%M:%S"))
        results_file = os.path.join(self.resultsDir, "HandCraftedFeatures"+now+".json")
        json.dump(self.Outputs, codecs.open(results_file, 'w', encoding='utf-8'))
        return super().close()
