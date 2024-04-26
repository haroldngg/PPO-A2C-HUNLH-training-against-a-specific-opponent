# PPO-A2C-HUNLH-training-against-a-specific-opponent

This repository contains adaptations and enhancements to the PettingZoo Poker environment, inspired by and building upon the foundational work by Chrys Rooper.
Source : https://github.com/curvysquare/PPO-and-A2C-for-HULH-poker

# Project Structure
* nolimitholdem_mod.py: This file includes modifications to the original PettingZoo environment and demonstrates the use of wrappers to enhance functionality.
* limitholdem/player.py: Located within the rlcard/game subdirectory, this file defines the action policies based on the behavior modeled from the initial player configuration.
* Model Locations:
  * PPO Model: The trained Proximal Policy Optimization model can be found in the logs2 directory.
  * A2C Model: The trained Advantage Actor-Critic model is located in the log_A2C directory.
 

The calculations of hand_strength and hand potential are based on https://github.com/Koda7/Poker-Analytics

