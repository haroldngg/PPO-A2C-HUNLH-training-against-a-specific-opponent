from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from rlcard.agents import DQNAgent
from rlcard.utils.utils import print_card
from treys import Evaluator
from treys import Card
import random
from keras.models import load_model

from deuces import Card, deck
from hp import HandPotential_1
from hse import hse_1

class PlayerStatus(Enum):
    ALIVE = 0
    FOLDED = 1
    ALLIN = 2
    
def format_legal_actions(legal_actions):
    inputs = legal_actions
    for i in legal_actions:
        if i == 'call':
            inputs[inputs.index('call')] = 'call, (0)'
        
        if i == 'raise':
            inputs[inputs.index('raise')] = 'raise, (1)'
        
        if i == 'fold':
            inputs[inputs.index('fold')] = 'fold, (2)'
        
        if i == 'check':
            inputs[inputs.index('check')] = 'check, (3)'
    return inputs

def _print_state(state):
    ''' Print out the state

    Args:
        state (dict): A dictionary of the raw state
        action_record (list): A list of the each player's historical actions
    '''
    state = state['raw_obs']
    print('\n=============== Community Card ===============')
    print_card(state['public_cards'])
    print('===============   Your Hand    ===============')
    print_card(state['hand'])

    print('===============     Pot      ===============')
    print(state['pot'])
    print('===============     Chips      ===============')
    print('Your chips:   ', end='')
    for _ in range(state['my_chips']):
        print('+', end='')
    print('')    
    print('Opponent chips:'  , end='')
    for _ in range(state['all_chips'][1]):
        print('+', end='')
    print('\n=========== Actions You Can Choose ===========')
    
    print(format_legal_actions(state['legal_actions']))
  

class LimitHoldemPlayer:
    """
        Attributes:
        np_random (np.random.RandomState): The random number generator.
        player_id (str): The identifier for the player.
        hand (list): The player's hole cards.
        status (PlayerStatus): The player's status (ALIVE, FOLDED, etc.).
        policy (str): The player's policy (e.g., 'random', 'PPO', 'human', 'heuristic').
        model: The player's machine learning model (if applicable).
        env: The game environment.
        rewardz (list): List to store rewards obtained during gameplay.
        opt_acts (list): List to store optimal actions taken by the player.
        in_chips (int): The chips that this player has put in the pot so far.

        Methods:
        get_state: Encode the state for the player
        get_player_id: Get the id of the player
        apply_mask: Apply a mask to a list
        get_action_mask: Get the action mask for the player
        get_action: Get the action for the player based off its policy attribute
    """

    def __init__(self, player_id, np_random, policy):
        """
        Initialize a player.

        Args:
            player_id (int): The id of the player
            np_random (np.random.RandomState): The random number generator
            policy (str): The policy of the player
        """
        self.np_random = np_random
        self.player_id = f"player_{player_id}"
        self.hand = []
        self.status = PlayerStatus.ALIVE
        self.policy = policy 
        self.model = None
        self.model_preflop = load_model(r"C:\Users\ngoup\Downloads\Modèle v1\modelepreflop.keras")
        self.model_flop = load_model(r"C:\Users\ngoup\Downloads\Modèle v1\modele_flop.keras")
        self.model_turn = load_model(r"C:\Users\ngoup\Downloads\Modèle v1\modele_turn.keras")
        self.model_river = load_model(r"C:\Users\ngoup\Downloads\Modèle v1\modele_river.keras")
        self.env = None
        self.rewardz = []
        self.opt_acts = []
        # The chips that this player has put in until now
        self.in_chips = 0

    def get_state(self, public_cards, all_chips, legal_actions):
        """
        Encode the state for the player

        Args:
            public_cards (list): A list of public cards that seen by all the players
            all_chips (int): The chips that all players have put in

        Returns:
            (dict): The state of the player
        """
        return {
            'hand': [c.get_index() for c in self.hand],
            'public_cards': [c.get_index() for c in public_cards],
            'all_chips': all_chips,
            'my_chips': self.in_chips,
            'legal_actions': legal_actions
        }

    def get_player_id(self):
        return self.player_id
    
    def apply_mask(initial_list, mask):
        masked_list = [value for value, valid in zip(initial_list, mask) if valid]
        return masked_list

    def get_action_mask(self):
        env = self.env
        obs = env.observe(self.player_id)
        if len(obs) > 2:
            mask = obs[0]['action_mask']
        else:
            mask = obs['action_mask'] 
        
        return mask
 
    def get_action(self, player_obs, game):
        """
        Get the action to be taken by the player based on their policy.

        Args:
            player_obs (dict or tuple): The observation of the player, containing relevant game information.
            game: The game environment.

        Returns:
            action: The selected action according to the player's policy.  
        """
        from rlcard.games.nolimitholdem.round import Action
        from rlcard.games.nolimitholdem.game import Stage

        def convert_card_format(cards):
            # Dictionnaire pour convertir les symboles de couleur
            suit_map = {
                'D': 'd',  # Diamonds
                'C': 'c',  # Clubs
                'H': 'h',  # Hearts
                'S': 's'   # Spades
            }
            
            # Convertir chaque carte dans le format souhaité
            converted_cards = [card[1] + suit_map[card[0]] for card in cards]
            return converted_cards
            
        def hand_strength_potential(hand, board):
            h = [Card.new(carte) for carte in hand]
            b = [Card.new(carte) for carte in board]
            if len(b)<5:
                try:
                    hpp, hpn = HandPotential_1(b, h)
                except:
                    print(board, hand)
            else:
                hpp, hpn = 0,0
            return (hse_1(b,h),hpp,hpn)

        def indice_middle(board_flop):
                sorted_flop = sorted(board_flop)
                return sorted_flop[1]

        def enlever_minimum(liste):
                    if not liste:
                        return []  # Retourne une liste vide si la liste est déjà vide
                    
                    minimum = min(liste)  # Trouve le minimum dans la liste
                    liste.remove(minimum)  # Enlève la première occurrence du minimum trouvé
                    return liste

        def valeur_cartes(joueur_cartes):
    # Tronquer le tableau à une longueur de 52
                joueur_cartes = joueur_cartes[:52]
    
                cartes = 0
                hauteurs_couleurs = []
                indice = []
    # Parcourir les cartes
                for i in range(52):
                    if joueur_cartes[i] == 1:
                        cartes += 1
            # Ajouter la position de la carte dans le tableau initial à la liste
                        indice.append(i)
                        hauteur = i % 13
                        hauteurs_couleurs.append(hauteur)
            # Si on a trouvé les deux cartes recherchées, sortir de la boucle
                    if cartes == 2:
                        break
    
    # Vérifier si les indices i et j sont dans la même plage de valeurs
                plages_couleurs = [(0, 12), (13, 25), (26, 38), (39, 51)]
                for plage in plages_couleurs:
                    if indice[0] in range(plage[0], plage[1]+1) and indice[1] in range(plage[0], plage[1]+1):
                        hauteurs_couleurs.append(0)  # Indique que les cartes sont de la même couleur
                        break
                    else:
                        hauteurs_couleurs.append(1)  # Indique que les cartes ne sont pas de la même couleur
    
                return hauteurs_couleurs

        if self.policy == 'Harold':

            if game.stage == Stage.PREFLOP:   # Preflop
                
                stack = player_obs['observation'][52]
                if game.dealer_id == 0:
                    position = 0
                else : 
                    position = 1
                
                nb_player_start = 2
                
                pot_size_before_decision = game.get_state(0)['pot']
                nb_unfolded_players = 2
                money_to_add_to_call = game.players[0].money_bet  -  game.players[1].money_bet
                last_raise__preflop_player = game.last_raiser_preflop

                top_card = max(valeur_cartes(player_obs['observation'])[:2])
                bottom_card = min(valeur_cartes(player_obs['observation'])[:2])
                color = 1 - valeur_cartes(player_obs['observation'])[2]

                X = [stack, position, nb_player_start, pot_size_before_decision, nb_unfolded_players, money_to_add_to_call, last_raise__preflop_player,top_card,bottom_card,color]
                X = np.array(X).reshape(1, -1).astype(np.float32)
                Y = self.model_preflop.predict(X)
                predicted_action = np.argmax(Y)

                action_map = {
                    0: Action.FOLD,
                    1: Action.CHECK_CALL,
                    2: Action.RAISE_HALF_POT
                }
                print("Preflop")
                action = action_map.get(predicted_action, None).value
        

            elif game.stage == Stage.FLOP:

                def cartes_flop(joueur_cartes):
                    # Tronquer le tableau à une longueur de 52
                    joueur_cartes = joueur_cartes[:52]
                    
                    cartes = 0
                    board_flop= []
                    # Parcourir les cartes
                    for i in range(52):
                        if joueur_cartes[i] == 2:
                            cartes += 1
                        
                            
                            carte = i % 13
                            board_flop.append(carte)
                            # Si on a trouvé les trois cartes recherchées, sortir de la boucle
                            if cartes == 3:
                                break
                    

                    return board_flop
                
                def color_flop (joueur_cartes):
    # Tronquer le tableau à une longueur de 52
                    joueur_cartes = joueur_cartes[:52]
                    
                    cartes = 0
                    flop_board = []
                    indice = []
                    # Parcourir les cartes
                    for i in range(52):
                        if joueur_cartes[i] == 2:
                            cartes += 1
                            # Ajouter la position de la carte dans le tableau initial à la liste
                            indice.append(i)
                            carte = i % 13
                            flop_board.append(carte)
                            # Si on a trouvé les deux cartes recherchées, sortir de la boucle
                            if cartes == 3:
                                break
                    
                    # Vérifier si les 3 couleurs sont distinctes, pareils ou si 2/3 sont de la même couleur
                    plages_couleurs = [(0, 12), (13, 25), (26, 38), (39, 51)]
                    for plage in plages_couleurs:
                        if indice[0] in range(plage[0], plage[1]+1) and indice[1] in range(plage[0], plage[1]+1)and indice[2] in range(plage[0], plage[1]+1):
                            return 0  # Indique que les cartes sont de la même couleur
                            
                        elif (indice[0] in range(plage[0], plage[1]+1) and indice[1] in range(plage[0], plage[1]+1)) or \
                    (indice[0] in range(plage[0], plage[1]+1) and indice[2] in range(plage[0], plage[1]+1)) or \
                    (indice[1] in range(plage[0], plage[1]+1) and indice[2] in range(plage[0], plage[1]+1)):
                            return 1

                    return 2


                if game.dealer_id == 0:
                    position = 0
                else : 
                    position = 1

                stack = player_obs['observation'][52]
                flop_position = position
                nb_player_start = 2
                pot_size_before_decision = game.get_state(0)['pot']
                nb_unfolded_players = 2
                money_to_add_to_call = game.players[0].money_bet - game.players[1].money_bet
                
                resultat = hand_strength_potential(convert_card_format(game.get_state(0)['hand']), convert_card_format(game.get_state(0)['public_cards']))


                hand_strength = resultat[0]
                hand_pos_potential =resultat[1]
                hand_neg_potential = resultat[2]


                last_raise__preflop_player = game.last_raiser_preflop
                last_raise_flop_player = game.last_raiser_flop
                
                top_card_player = max(valeur_cartes(player_obs['observation'])[:2])
                diff_between_cards_player =top_card_player -  min(valeur_cartes(player_obs['observation'])[:2])

                sort = sorted(cartes_flop(player_obs['observation']))
                diff_2_3_highest_board = sort[1] - sort[0]
                diff_1_2_highest_board = sort[2] - sort[1]
                
                top_card_board = max(cartes_flop(player_obs['observation']))
                color = color_flop(player_obs['observation'])



                X = [stack, flop_position, nb_player_start, pot_size_before_decision, nb_unfolded_players, money_to_add_to_call,hand_strength, hand_pos_potential, hand_neg_potential, last_raise__preflop_player,last_raise_flop_player,top_card_player,diff_between_cards_player,  diff_2_3_highest_board, diff_1_2_highest_board, top_card_board,color]
                X = np.array(X).reshape(1, -1).astype(np.float32)
                Y = self.model_flop.predict(X)
                predicted_action = np.argmax(Y)

                action_map = {
                    0: Action.FOLD,
                    1: Action.CHECK_CALL,
                    2: Action.RAISE_HALF_POT
                }
                mask1 = player_obs['action_mask']
               # print(mask1)
               # print(action_map.get(predicted_action, None))
                print("Flop")
                action = action_map.get(predicted_action, None).value

        
            elif game.stage == Stage.TURN:   # turn
                def cartes_turn(joueur_cartes):
                    # Tronquer le tableau à une longueur de 52
                    joueur_cartes = joueur_cartes[:52]
                    
                    cartes = 0
                    board_turn= []
                    # Parcourir les cartes
                    for i in range(52):
                        if joueur_cartes[i] == 2:
                            cartes += 1
                        
                            
                            carte = i % 13
                            board_turn.append(carte)
                            # Si on a trouvé les trois cartes recherchées, sortir de la boucle
                            if cartes == 4:
                    
                                break
                        
                    return board_turn

                

                
                def turn_color(joueur_cartes):
                    # Tronquer le tableau à une longueur de 52
                    joueur_cartes = joueur_cartes[:52]
                    
                    cartes = 0
                    turn_board = []
                    indices = []
                    # Parcourir les cartes
                    for i in range(52):
                        if joueur_cartes[i] == 2:
                            cartes += 1
                            # Ajouter la position de la carte dans le tableau initial à la liste
                            indices.append(i)
                            carte = i % 13
                            turn_board.append(carte)
                            # Si on a trouvé les quatre cartes recherchées, sortir de la boucle
                            if cartes == 4:
                                break
                    
                    # Vérifier les différents types de flop
                    colors = [(0, 12), (13, 25), (26, 38), (39, 51)]
                    same_kind = False
                    same_color_count = 0
                    for color in colors:
                        color_indices = [idx for idx in indices if idx in range(color[0], color[1] + 1)]
                        if len(color_indices) == 4:
                            return 4  # 4 cartes de la même couleur (flush)
                        elif len(color_indices) == 3:
                            same_color_count += 1
                    if same_color_count == 3:
                        return 3  # 3 cartes de la même couleur et 1 d'une autre couleur (split 3/1)
                    
                    # Vérifier les autres types de splits
                    unique_cards = len(set(turn_board))
                    if unique_cards == 2:
                        return 2  # 2/2 split
                    elif unique_cards == 3:
                        return 1  # 2/1/1 split
                    else:
                        return 0  # 1/1/1/1 split

                stack = player_obs['observation'][52]
                if game.dealer_id == 0:
                    position = 0
                else : 
                    position = 1
                turn_position = position
                nb_player_start = 2
                pot_size_before_decision = game.get_state(0)['pot']
                nb_unfolded_players = 2
                money_to_add_to_call = game.players[0].money_bet - game.players[1].money_bet
                resultat = hand_strength_potential(convert_card_format(game.get_state(0)['hand']), convert_card_format(game.get_state(0)['public_cards']))


                hand_strength = resultat[0]
                hand_pos_potential = resultat[1]
                hand_neg_potential = resultat[2]


                position_last_raise__flop_player = game.last_raiser_flop
                position_last_raise_turn_player = game.last_raiser_turn

                top_card_player = max(valeur_cartes(player_obs['observation'])[:2])
                diff_between_cards_player =top_card_player -  min(valeur_cartes(player_obs['observation'])[:2])

                sort = sorted(cartes_turn(player_obs['observation']))
                diff_2_3_highest_board = sort[2] - sort[1]
                diff_1_2_highest_board = sort[3] - sort[2]
                diff_3_4_highest_board = sort[1] - sort[0]
                top_card_board = max(cartes_turn(player_obs['observation']))

                color = turn_color(player_obs['observation'])

                X = [stack, turn_position, nb_player_start, pot_size_before_decision, nb_unfolded_players, money_to_add_to_call,hand_strength, hand_pos_potential, hand_neg_potential,  position_last_raise__flop_player,position_last_raise_turn_player,top_card_player,diff_between_cards_player ,diff_2_3_highest_board,diff_1_2_highest_board, diff_3_4_highest_board, top_card_board,color]
                X = np.array(X).reshape(1, -1).astype(np.float32)
                Y = self.model_turn.predict(X)
                predicted_action = np.argmax(Y)
                            
                action_map = {
                    0: Action.FOLD,
                    1: Action.CHECK_CALL,
                    2: Action.RAISE_HALF_POT
                }
                mask1 = player_obs['action_mask']
               # print(mask1)
               # print(action_map.get(predicted_action, None))
                print("Turn")
                action = action_map.get(predicted_action, None).value

            elif game.stage == Stage.RIVER:   # river

                def cartes_river(joueur_cartes):
                    # Tronquer le tableau à une longueur de 52
                    joueur_cartes = joueur_cartes[:52]
                    cartes = 0
                    board_river= []
                    # Parcourir les cartes
                    for i in range(52):
                        if joueur_cartes[i] == 2:
                            cartes += 1
                        
                            
                            carte = i % 13
                            board_river.append(carte)
                            # Si on a trouvé les cinq cartes recherchées, sortir de la boucle
                            if cartes == 5:
                    
                                break
                        
                    return board_river




                def river_analysis(joueur_cartes):
                    # Tronquer le tableau à une longueur de 52
                    joueur_cartes = joueur_cartes[:52]
                    
                    cartes = 0
                    river_board = []
                    indices = []
                    # Parcourir les cartes
                    for i in range(52):
                        if joueur_cartes[i] == 2:
                            cartes += 1
                            # Ajouter la position de la carte dans le tableau initial à la liste
                            indices.append(i)
                            carte = i % 13
                            river_board.append(carte)
                            # Si on a trouvé les cinq cartes recherchées, sortir de la boucle
                            if cartes == 5:
                                break
                    
                    # Vérifier les différents types de river
                    unique_cards = len(set(river_board))
                    if unique_cards == 1:
                        return 5  # 5/0/0/0
                    elif unique_cards == 2:
                        color_counts = [0] * 4
                        for idx in indices:
                            for i, color_range in enumerate([(0, 12), (13, 25), (26, 38), (39, 51)]):
                                if idx in range(color_range[0], color_range[1] + 1):
                                    color_counts[i] += 1
                                    break
                        if 4 in color_counts:
                            return 4  # 4/1/1/1
                        elif 3 in color_counts and 2 in color_counts:
                            return 3  # 3/2/0/0
                        elif color_counts.count(3) == 1 and color_counts.count(2) == 1:
                            return 2  # 3/1/1/0
                        elif color_counts.count(2) == 2 and color_counts.count(3) == 0:
                            return 1  # 2/2/1/0
                    elif unique_cards == 4:
                        return 0  # 2/1/1/1
                    return -1 # Cas non géré

                
            
                stack = player_obs['observation'][52]
                if game.dealer_id == 0:
                    position = 0
                else : 
                    position = 1
                river_position = position
                nb_player_start = 2
                pot_size_before_decision = game.get_state(0)['pot']
                nb_unfolded_players = 2
                money_to_add_to_call = game.players[0].money_bet - game.players[1].money_bet
                resultat = hand_strength_potential(convert_card_format(game.get_state(0)['hand']), convert_card_format(game.get_state(0)['public_cards']))


                hand_strength = resultat[0]


                position_last_raise_turn_player = game.last_raiser_turn
                position_last_raise_river_player = game.last_raiser_river

                top_card_player = max(valeur_cartes(player_obs['observation'])[:2])
                diff_between_cards_player =top_card_player -  min(valeur_cartes(player_obs['observation'])[:2])

                sort = sorted(cartes_river(player_obs['observation']))
                diff_1_2_highest_board = sort[4] - sort[3]
                diff_2_3_highest_board = sort[3] - sort[2]
                diff_3_4_highest_board = sort[2] - sort[1]
                diff_4_5_highest_board = sort[1] - sort[0]
                top_card_board = max(cartes_river(player_obs['observation']))

                
                color = river_analysis(player_obs['observation'])

                X = [stack, river_position, nb_player_start, pot_size_before_decision, nb_unfolded_players, money_to_add_to_call,hand_strength, position_last_raise_turn_player,position_last_raise_river_player,top_card_player,top_card_board,diff_between_cards_player,diff_1_2_highest_board, diff_2_3_highest_board, diff_3_4_highest_board, diff_4_5_highest_board, color]
                X = np.array(X).reshape(1, -1).astype(np.float32)
                Y = self.model_river.predict(X)
                predicted_action = np.argmax(Y)
                            
                action_map = {
                    0: Action.FOLD,
                    1: Action.CHECK_CALL,
                    2: Action.RAISE_HALF_POT
                }
                mask1 = player_obs['action_mask']
               # print(mask1)
               # print(action_map.get(predicted_action, None))
                print("river")
                action = action_map.get(predicted_action, None).value




        if self.policy == 'random':
            mask1 = player_obs['action_mask']
            action = self.env.action_space(self.player_id).sample(mask1)

        if self.policy == 'PPO' or self.policy == 'A2C':
            if type(player_obs) == tuple:
                player_obs = player_obs[0]
                pass
            action = self.model.predict(observation=player_obs)
            action = action[0]
                       
        if self.policy == 'human':
            raw_env = self.env.env.env.env
            limit_holdem_env = self.env.env.env.env.env
            state  = limit_holdem_env.get_state(0)
            print(_print_state(state))
            try:
                action = int(input('>> You choose action (integer): '))
            except ValueError:
                action = int(input('>> You choose action (integer): '))
            action = action    
            
        if self.policy == 'heuristic':
           action = self.optimal_action(player_obs, game)
           if not self.check_for_one_at_index(player_obs['action_mask'], action):
            action = self.env.action_space(self.player_id).sample(player_obs['action_mask'])

        return action 

    def check_for_one_at_index(self, my_array, selected_index):
    # Check if the selected ction is valid
        if 0 <= selected_index < len(my_array):
            # Check if there is a 1 at the selected index
            if my_array[selected_index] == 1:
                return True
            else:
                return False
        else:
            return False                     
    def optimal_action(self, player_obs, game):
        """

        Determine the optimal action for the player 

        This method calculates theplayer's optimal action based on their current hand, the public cards in the game, and predefined quartile scores.
        If there are at least three public cards, it evaluates the hand's strength and selects an action based on quartile score ranges.
        If there are fewer than three public cards, it selects a random action from the available actions according to the provided action mask.

        Parameters:
        - player_obs (dict): A dictionary containing observation information for the AI player.
        - game (Game): The current poker game being played.

        Returns:
        - op_act (int): The selected optimal action for the AI player. Possible values:
            - 0: call
            - 1: raise 
            - 2: fold
            - 3: check
         """
        score_max = 7462
        quartiles = [score_max * 0.25, score_max * 0.5, score_max * 0.75]

        hand = []
        for c in self.hand:
            c1r = c.rank
            c1s = c.suit.lower()
            c1 = c1r +  c1s
            hand.append(c1)
  
        pc = []
        if len(game.public_cards) > 0:
            public_cards = game.public_cards
                     
            for c in public_cards:
                cr_temp = c.rank
                cs_temp = c.suit.lower()
                pc.append(cr_temp +  cs_temp)
                
        hand_objs = []
        pc_objs = [] 
        for c in hand:
            hand_objs.append(Card.new(c))   
        for c in pc:
            pc_objs.append(Card.new(c))       
        
        if len(pc) >= 3:
            evaluator = Evaluator()
            try: 
                score = evaluator.evaluate(hand_objs, pc_objs)
            except:
                KeyError
                score = 0
            
            if score <= quartiles[0]:
                op_act = 1
            if score >= quartiles[0] and score <= quartiles[1]:
                op_act = 0
            if score >= quartiles[1] and score <= quartiles[2]:
                op_act = 3
            if score >= quartiles[2]:
                op_act = 2  
        else:            
            mask1 = player_obs['action_mask']
            legal_acts =  [i for i, value in enumerate(mask1) if value == 1]
            op_act = random.choice(legal_acts)
      
            
        return op_act
  
               
        
                 
                
        

