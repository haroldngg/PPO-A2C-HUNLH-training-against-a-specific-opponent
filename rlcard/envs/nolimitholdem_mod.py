import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.nolimitholdem import Game
from rlcard.games.nolimitholdem.round import Action
from rlcard.games.nolimitholdem.round import NolimitholdemRound
from rlcard.games.limitholdem.player import LimitHoldemPlayer
from enum import Enum


DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        'chips_for_each': 100,
        'dealer_id': None,
        }

class NolimitholdemEnv(Env):
    ''' Limitholdem Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.round = NolimitholdemRound(2,0,0,0)
        self.name = 'no-limit-holdem-mod'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = Action
        self.obs_shape = None
        self.action_shape = [None for _ in range(self.num_players)]
        
        
        # for raise_amount in range(1, self.game.init_chips+1):
        #     self.actions.append(raise_amount)

        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)


    def set_state_shape(self, obs_shape, obs_type):
        """
        set the state shape, ie the boolean vector length depending on the observation type. 

        Args:
            obs_shape (_type_): boolean vector length
            obs_type (_type_): observation type (54)
        """
        self.obs_shape = obs_shape 
        self.obs_type = obs_type
        # set the state shape, ie the boolean vector length depending on the observation type.       

        if self.obs_shape[0] == '54' and self.obs_type == '54':
            self.state_shape = [[56] for _ in range(self.num_players)]
        
        if self.obs_shape[0] == '55' and self.obs_type == '55':
            self.state_shape = [[55] for _ in range(self.num_players)]


    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            extracted_state (dict): A dictionary containing the extracted state representation.
                - 'legal_actions' (OrderedDict): Legal actions available to the agent.
                - 'obs' (numpy.ndarray): The observation array based on the specified observation shape and type.
                - 'raw_obs' (dict): The raw state observation from the game.
                - 'raw_legal_actions' (list): List of raw legal actions.
                - 'action_record' (ActionRecorder): The action recorder for the agent.
        '''
        extracted_state = {}
        #print("Starting state extraction...")
    # Ajout de vérifications et d'impressions de debug
       # print("Current hand:", state.get('hand', []))
       # print("Current round:" ,state.get('stage'))
       # print("public cards : ", state.get('public_cards'), [])
       # print("last opp action : ", state.get('opponent_last_action', None))
       # print("Mes chips :", state.get('my_chips', 0) )
        if self.obs_shape is None:
            self.obs_shape = [56]

       # legal_actions = OrderedDict({action.value: None for action in state['legal_actions']})
        legal_actions = OrderedDict((action.value, None) if isinstance(action, Enum) else (action, None) for action in state['legal_actions'])

        extracted_state['legal_actions'] = legal_actions

        public_cards = state.get('public_cards', [])
        hand = state.get('hand', [])
        my_chips = state.get('my_chips', 0)
        all_chips = state.get('all_chips', [])
        tour = state.get('stage', None).value if state.get('stage') else 0
        last_opp_action = state.get('opponent_last_action', None)
        last_opp_action = self.round.get_last_action((self.get_player_id() + 1) % self.num_players)
        if last_opp_action is not None:
            last_opp_action = last_opp_action.value
        else:
            last_opp_action = 0

# first deck of cards indexes the player cards, second indexes the community cards.

        if self.obs_shape[0] == '54' and self.obs_type == '54':
            hand_idx = [self.card2index[card] for card in hand]
            public_cards_idx = [self.card2index[card] for card in public_cards]
            obs = np.zeros(56)
            obs[hand_idx] = 1
            obs[:52][public_cards_idx] = 2
            obs[52] = float(my_chips)
            if all_chips:
                max_chips = float(max(all_chips))
            else:
                max_chips = 0.0
            obs[53] = float(max_chips)
            obs[54] = float(tour)
            obs[55] = float(last_opp_action)
            extracted_state['obs'] = obs

            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
            extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions(action_id) not in legal_actions:
            if Action.CHECK in legal_actions:
                return Action.CHECK
            else:
                print("Tried non legal action", action_id, self.actions(action_id), legal_actions)
                return Action.FOLD
        return self.actions(action_id)

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = [c.get_index() for c in self.game.public_cards] if self.game.public_cards else None
        state['hand_cards'] = [[c.get_index() for c in self.game.players[i].hand] for i in range(self.num_players)]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state


