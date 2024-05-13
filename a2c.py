from stable_baselines3 import A2C

import torch as th
from torch import nn

import numpy as np 
import os

import nolimit_texas_holdem_mod 
from rlcard_envs.rlcard_base_mod import RLCardBase

import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import EvalCallback

device = th.device("cuda" if th.cuda.is_available() else "cpu")
log_dir = "./log_A2C"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
env = nolimit_texas_holdem_mod.env(obs_type='54',render_mode=None)
env.OPPONENT.policy = "Harold"

#print("Output of get_state:", obs)


eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=1500,
                             deterministic=True, render=False,)

model2 = A2C('MultiInputPolicy', env, n_steps=15, verbose=1, tensorboard_log=log_dir, device=device)
model2.learn(total_timesteps=150000, callback=eval_callback, progress_bar= True, tb_log_name="A2C")

model2.save(r"/users/eleves-a/2022/noam-joud-harold.ngoupeyou/a2c.zip")
