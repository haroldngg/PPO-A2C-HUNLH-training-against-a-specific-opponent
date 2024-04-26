#from pettingzoo.classic import texas_holdem_no_limit_v6
from pettingzoo.test import test_save_obs
import nolimit_texas_holdem_mod 
from rlcard_envs.rlcard_base_mod import RLCardBase
from pettingzoo.test import api_test
from pettingzoo.test import test_save_obs
from pettingzoo.butterfly import cooperative_pong_v5
from stable_baselines3 import PPO
import torch as th
from torch import nn

import numpy as np 
import os

import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import EvalCallback

"""
TRAINING DU MODELE 


device = th.device("cuda" if th.cuda.is_available() else "cpu")
log_dir = "./log2"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
env = nolimit_texas_holdem_mod.env(obs_type='54',render_mode=None)
env.OPPONENT.policy = "Harold"

#print("Output of get_state:", obs)


eval_callback = EvalCallback(env, best_model_save_path="./logs2/",
                             log_path="./logs2/", eval_freq=2048,
                             deterministic=True, render=False,)


model = PPO("MultiInputPolicy", env , verbose=1, device=device, tensorboard_log= log_dir )
model.learn(total_timesteps=150000, callback=eval_callback, progress_bar= True, tb_log_name="PPO")
  

# Sauvegarder le modèle
model.save(r"/users/eleves-a/2022/noam-joud-harold.ngoupeyou/ppo.zip")

"""
#### TEST du modèle

num_episodes = 1000
env = nolimit_texas_holdem_mod.env(obs_type='54',render_mode='Human')
env.OPPONENT.policy = "Harold"

model = PPO.load(r"C:\Users\ngoup\Downloads\PSC_verif\PSC_2\logs2\best_model.zip", env = env)
total_rewards = []
c  = 0

def contains_second_dict(data):
    if isinstance(data, (list, tuple)) and len(data) > 1:
        return isinstance(data[1], dict)
    return False

for episode in range(num_episodes):
        observation = env.reset()
        env.render()
        

        done = False
        total_reward = 0
        
        
        while not done:
            
            result = contains_second_dict(observation)

            if result == True :
                action, _ = model.predict(observation[0])
            else:
                action, _ = model.predict(observation)

            print(action)
            observation, reward, done, info, _ = env.step(action)
            
            total_reward += reward
        
        if(total_reward > 0):
            c += 1 

        total_rewards.append(total_reward)
        print(f"Épisode {episode + 1}: Récompense Totale = {total_reward}")
    
average_reward = sum(total_rewards) / num_episodes
print(f"Récompense moyenne sur {num_episodes} épisodes: {average_reward}")
print(f"Notre agent il a gagné le joueur durant {c} parties sur {num_episodes}")
