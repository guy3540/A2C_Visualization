import gym
from HCF_functions import get_paddle_position, get_max_tunnel_depth, get_ball_position
from HCF_gym_wrapper import HCFgymWrapper
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C


NUM_TIMESTEPS = 50000
NUM_OF_PROCESSES = 16

# Parallel environments
env = make_vec_env('Breakout-v4', n_envs=NUM_OF_PROCESSES, wrapper_class=HCFgymWrapper)

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=NUM_TIMESTEPS)
model.save("a2c_Breakout-v4")
model.env.close()

del model  # remove to demonstrate saving and loading

model = A2C.load("a2c_Breakout-v4")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
