
import numpy as np   
from pettingzoo.classic.rlcard_envs import texas_holdem
from pettingzoo.classic.rlcard_envs import nolimitholdem_mod
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd

import os 
import matplotlib.pyplot as plt

from gymnasium import Env
import optuna
import gym
import numpy as np
import torch as th
from torch import nn
from tabulate import tabulate
import pandas as pd
#from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent

from injector import card_injector
from stable_baselines3.common.env_checker import check_env

env = nolimit_texas_holdem_mod.env(obs_type='106',render_mode="human")
env.AGENT.policy = 'random'
env.OPPONENT.policy = 'random'
# It will check your custom environment and output additional warnings if needed
check_env(env)