from collections import defaultdict
import math
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
import time
import heapq
from seahorse.game.light_action import LightAction
from board_divercite import BoardDivercite
from typing import Dict

MAX_DEPTH = 40
TWO_PLY_STEP = 2
CITIES_POS = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5), (7, 4)]
RESSOURCES_POS = [(0, 4), (1, 3), (1, 5), (2, 2), (2, 4), (2, 6), (3, 1), (3, 3), (3, 5), (3, 7), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8), (5, 1), (5, 3), (5, 5), (5, 7), (6, 2), (6, 4), (6, 6), (7, 3), (7, 5), (8, 4)]
class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "Oppy"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        self._transposition_table = {}
        self._timeout = 0
        self._number_of_remaining_moves = 20
        self._time_allocation = {
            20:0.2,
            19:0.2,
            18:0.3,
            17:0.4,
            16:0.6,
            15:0.7,
            14:1.1,
            13:1.2,
            12:1.3,
            11:1.6,
            10:1.55,
            9:1.45,
            8:1.45,
            7:1.45,
            6:1.2,
            5:0.1,
            4:0.05,
            3:0.05,
            2:0.05,
            1:0.05
          }
        self.phase = 'EARLY'

    def _handle_phase_change(self, state: GameState):
        number_remaining_pieces = sum([value for value in state.players_pieces_left[self.get_id()].values()])
        if 16 < self._number_of_remaining_moves:
            self.phase = 'EARLY'
        elif 7 < self._number_of_remaining_moves :
            self.phase = 'MID'
        else:
            self.phase = 'LATE'

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.
        We use two-ply iterative deepening to improve the efficiency of the minimax algorithm.
        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        self._handle_phase_change(current_state)
    

        action = None
        self._timeout = time.time() + (self._time_allocation[ self._number_of_remaining_moves] * 60)- 0.5
        start_time = time.time()

        for depth in range(1, MAX_DEPTH + 1, TWO_PLY_STEP):
            if time.time() > self._timeout:
                break
            _, action = self._minimax(current_state, depth, float('-inf'), float('inf'), maximizing=True)
        self._number_of_remaining_moves -= 1
        return action
    
    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing):
        self._handle_phase_change(state)
        if hash(state) in self._transposition_table:
            stored_depth, stored_score = self._transposition_table[hash(state)]['depth'], self._transposition_table[hash(state)]['score']
            if stored_depth >= depth: # TODO: check if we correctly compare the depth
                return stored_score, None

        if depth == 0 or state.is_done():
            score = self._evaluate_board(state)
            self._transposition_table[hash(state)] = {'depth': depth, 'score': score}
            return score, None
        
        bestMove = None
        if maximizing:
            maxEval = float('-inf')
            actions = self._generate_heap_light_action(state, is_max_heap=True)
            while len(actions) > 0 and time.time() < self._timeout:
                action = heapq.heappop(actions)[2]
                score, _ = self._minimax(action.get_heavy_action(state).get_next_game_state(), depth - 1, alpha, beta, False)
                if score > maxEval:
                    maxEval = score
                    bestMove = action
                    alpha = max(alpha, maxEval)

                if score >= beta:
                    maxEval = score
                    bestMove = action
                    break # pruning
            if bestMove is None:
                if len(actions) == 0:
                    actions = self._generate_heap_light_action(state, is_max_heap=True)
                    print()
                bestMove = heapq.heappop(actions)[2]
                maxEval = self._evaluate_board(bestMove.get_heavy_action(state).get_next_game_state())

            self._transposition_table[hash(state)] = {'depth': depth, 'score': maxEval}    
            return maxEval, bestMove
        
        else:
            minEval = float('inf')
            actions = self._generate_heap_light_action(state, is_max_heap=False)
            while len(actions) > 0 and time.time() < self._timeout:
                action = heapq.heappop(actions)[2]
                score, _ = self._minimax(action.get_heavy_action(state).get_next_game_state(), depth - 1, alpha, beta, True)
                if score < minEval:
                    minEval = score
                    bestMove = action

                    beta = min(beta, minEval)

                if minEval <= alpha:
                    minEval = score
                    bestMove = action
                    break # pruning
                    
            if bestMove is None:
                if len(actions) == 0:
                    actions = self._generate_heap_light_action(state, is_max_heap=True)
                    print()
                bestMove = heapq.heappop(actions)[2]
                minEval = self._evaluate_board(bestMove.get_heavy_action(state).get_next_game_state())
            
            self._transposition_table[hash(state)] = {'depth': depth, 'score': minEval}
            return minEval, bestMove

    def _get_opponent_id(self, state: GameState):
        keys = list(state.scores.keys())
        return (keys[0], keys[1]) if keys[0] == self.get_id() else (keys[1], keys[0])
    
    def _generate_heap_light_action(self, state: GameState, is_max_heap: bool) -> list:
        heap = []
        current_rep = state.get_rep()
        b = current_rep.get_env()
        d = current_rep.get_dimensions()
        free_positions = self._get_free_positions(b, d)
        multiplicator = -1 if is_max_heap else 1
        for piece, n_piece in state.players_pieces_left[state.next_player.get_id()].items():
            piece_color = piece[0]
            piece_res_city = piece[1]
            if n_piece > 0:
                for i, j in free_positions:
                    if state.in_board((i, j)) and (i,j) not in b and state.piece_type_match(piece_res_city, (i, j)):
                        data = {"piece": piece_color+piece_res_city, "position" : (i,j)}
                        action = LightAction(data)
                        new_state = state.apply_action(action)
                        score = self._evaluate_board(new_state)
                        heapq.heappush(heap, (multiplicator * score, id(action), action))
        return heap

    def _get_free_positions(self, b, d):
        board = BoardDivercite(env=b, dim=d)
        grid_data = board.get_grid()
        return [(i, j) for i, row in enumerate(grid_data) for j, cell in enumerate(row) if (cell == ('◇ ', 'Black') or cell == ('▢ ', 'Black'))]
    
    def _evaluate(self, state: GameState):
        _, id_opponent = self._get_opponent_id(state)
        return state.scores[self.get_id()] - state.scores[id_opponent]

    def _evaluate_board(self, state: GameState) -> int:
        if self.phase == 'EARLY':
            return self._evaluate_early(state)
        elif self.phase == 'MID':
            return self._evaluate_mid(state)
        else:
            return self._evaluate_late(state)

    def _evaluate_early(self, state: GameState) -> int:
        score = 0
        score += 10 * self._verify_variety(state)
        score += 10 * self._blocked_divercite(state)
        score += 5 * self._prioritize_cities(state)
        score += 1 * self._placement(state)
        score += 10 * self._evaluate(state)
        return score
    
    def _evaluate_mid(self, state: GameState) -> int:
        score = 0
        score += 35 * self._divercity_count(state)
        score += 5 * self._block_opponent_divercite(state)
        score += 10 * self._closeness_divercite(state)
        score += 5 * self._blocked_divercite(state)
        score += 15 * self._evaluate(state)
        return score

    def _evaluate_late(self, state: GameState) -> int:
        if state.is_done():
            scores = state.remove_draw(state.scores, state.get_rep())
            if scores[self.get_id()] > scores[self._get_opponent_id(state)[1]]:
                return 10000
            else:
                return -10000
        else:
            return self._evaluate(state) # TODO: add a better evaluation function

    def _verify_variety(self, state: GameState) -> int:
        """
        Between 
        """
        score = 0
        items = state.players_pieces_left[self.get_id()]

        filter_items = lambda item_type: {item: quantity for item, quantity in items.items() if item[1] == item_type}

        ressources = defaultdict(int, filter_items('R'))
        cities = defaultdict(int, filter_items('C')) 

        score += 10 / (self._variety_ressources(ressources) + 1)
        score += 5 / (self._variety_cities(cities) + 1)
        return score
    
    def _variety_ressources(self, ressources: Dict) -> int:
        """
        Evaluate the variety of ressources used by our player.

        Args:
            ressources (Dict): The ressources of the player.

        Returns:
            int: The variety of ressources.
        """
        ressources_count = [value for value in ressources.values()]
        min_count = min(ressources_count)
        max_count = max(ressources_count)
        return max_count - min_count
    
    def _variety_cities(self, ressources: Dict) -> int:
        """
        Evaluate the variety of cities used by our player.

        Args:
            ressources (Dict): The cities of the player.

        Returns:
            int: The variety of cities.
        """
        cities_count = [value for value in ressources.values()]
        min_count = min(cities_count)
        max_count = max(cities_count)
        return max_count - min_count
    
    def _blocked_divercite(self, state: GameState) -> int:
        """
        Make sure that we are not in a state where some of our cities cannot do a divercite

        Args:
            state (GameState): The current game state.
    
        Returns:
            int: The score of the evaluation
        """
        count = 0
        for pos in CITIES_POS:
            if state.get_rep().get_env().get(pos) is not None and not state.check_divercite(pos):
                owner_city = state.get_rep().get_env()[pos].piece_type[-1]
                if owner_city != self.piece_type:
                    continue
                neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
                neighboring_colors = set()
                for _, value in neighbors.items():
                    if value[0] != 'EMPTY':
                        if value[0] not in neighboring_colors:
                            neighboring_colors.add(value[0])
                        else:
                            count -= 1
        return count * 100

    def _prioritize_cities(self, state: GameState) -> int:
        """
        Prioritize placing cities before ressources

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score of the evaluation
        """

        items = state.players_pieces_left[self.get_id()]
        sum_items = lambda item_type: sum(quantity for item, quantity in items.items() if item[1] == item_type)

        total_ressources = 12 - sum_items('R')
        total_cities = 8 - sum_items('C')

        return total_cities - total_ressources

    def _placement(self, state: GameState) -> int:
        """
        Evaluate the placement of the cities

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score of the evaluation
        """
        score = 0
        score += self._cities_position(state)
        # score += self._spacing(state)
        return score
    
    def _cities_position(self, state: GameState) -> int:
        """
        Evaluate the position of the cities

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score of the evaluation
        """
        cities_pos = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5), (7, 4)]
        positons_score = {
            (1, 4): 10, (4, 2): 10, (4, 7): 10, (7, 4): 10, # Corners
            (3, 4): 10, (4, 3): 10, (4, 5): 10, (5, 4): 10, # Middle
            (2, 3): 5, (2, 5): 5, (3, 2): 5, (3, 6): 5, (5, 2): 5, (5, 6): 5, (6, 3): 5, (6, 5): 5 # Sides
        }
        score = 0
        for pos in cities_pos:
            if state.get_rep().get_env().get(pos) is not None:
                owner_city = state.get_rep().get_env()[pos].piece_type[-1]
                if owner_city != self.piece_type:
                    score -= positons_score.get(pos, 0)
                else:
                    score += positons_score.get(pos, 0)

        return score
    
    def _spacing(self, state: GameState) -> int:
        """
        Evaluate the proximitiy of our cities with all of the others

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score of the evaluation
        """
        cities_pos = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5), (7, 4)]
        score = 0
        for pos in cities_pos:
            if state.get_rep().get_env().get(pos) is not None:
                owner_city = state.get_rep().get_env()[pos].piece_type[-1]
                if owner_city != self.piece_type:
                    continue
                neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
                neighboring_colors = set()
                for neighbor, value in neighbors.items():
                    if value[0] != 'EMPTY':
                        if value[0] not in neighboring_colors:
                            neighboring_colors.add(value[0])
                        else:
                            score -= 5

    def _divercity_count(self, state: GameState) -> int:
        """
        Evaluate the number of divercite for each player.

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score of the evaluation
        """
        player_id, opponent_id =  self._get_opponent_id(state)
        divercite = {
            player_id: 0,
            opponent_id: 0
        }
        for pos in CITIES_POS:
            if state.get_rep().get_env().get(pos) is not None and state.check_divercite(pos):
                owner_city = state.get_rep().get_env()[pos].piece_type[-1]
                if owner_city == self.piece_type:
                    divercite[player_id] += 1
                else:
                    divercite[opponent_id] += 1

        # for i in range(state.get_rep().get_dimensions()[0]):
        #     for j in range(state.get_rep().get_dimensions()[1]):
                
        #         if state.in_board((i, j)) and state.piece_type_match('C', (i, j)) and (i, j) in state.get_rep().get_env() and state.check_divercite((i, j)):
        #             players_color = self.get_piece_type()
        #             city_color = state.get_rep().get_env()[(i, j)].piece_type[-1]
        #             if players_color == city_color:
        #                 divercite[player_id] += 1
        #             else:   
        #                 divercite[opponent_id] += 1
        score = divercite[player_id] - divercite[opponent_id]
        return score * 100

    def _block_opponent_divercite(self, state: GameState) -> int:
        """
        Evaluate the number of opponent's cities that cannot do divercite.

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score of the evaluation
        """
        player_id, opponent_id =  self._get_opponent_id(state)

        cities_blocked = self.cities_with_same_ressources(state)
        count_cities_opponent_blocked = cities_blocked[opponent_id][2]
        count_cities_blocked = cities_blocked[player_id][2] 

        return (count_cities_opponent_blocked - count_cities_blocked) * 100
    
    def cities_with_same_ressources(self, state: GameState) -> Dict[str, int]:
        """
        Evaluate how many cities have the same ressources.

        Args:
            state (GameState): The current game state.
    
        Returns:
            Dict[str, int]: The number of cities with the same ressources for each player.
        """
        player_id, opponent_id =  self._get_opponent_id(state)
        cities = {
            player_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            opponent_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        }
        players_color = self.get_piece_type()
        for pos in CITIES_POS:
            if state.get_rep().get_env().get(pos) is not None:
                city_color = state.get_rep().get_env()[pos].piece_type[0]
                neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
                count = 0
                for _, value in neighbors.items():
                    if value[0] != 'EMPTY' and value[0] == city_color:
                        count += 1
                if players_color == city_color:
                    cities[player_id][count] += 1
                else:
                    cities[opponent_id][count] += 1
        return cities
    
    def _closeness_divercite(self, state: GameState) -> int:
        possible_divercite = self.possible_divercite(state)
        player_id, opponent_id = self._get_opponent_id(state)
        return (possible_divercite[player_id][3] - possible_divercite[opponent_id][3]) * 100

    def possible_divercite(self, state: GameState) -> Dict[str, Dict]:
        """
        Evaluate how many cities are close to divercite and to what degree.

        Args:
            state (GameState): The current game state.
    
        Returns:
            Dict[str, Dict]: The number of cities close to divercite for each player.
        """
        player_id, opponent_id =  self._get_opponent_id(state)
        cities = {
            player_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            opponent_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        }
        
        players_color = self.get_piece_type()
        cities_pos = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5), (7, 4)]
        for pos in cities_pos:
            if state.get_rep().get_env().get(pos) is not None and not state.check_divercite(pos):
                color = state.get_rep().get_env()[pos].piece_type[0]
                neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
                neighboring_colors = set()
                for _, value in neighbors.items():
                    if value[0] != 'EMPTY':
                        if value[0] not in neighboring_colors:
                            neighboring_colors.add(value[0])
                        else:
                            continue
                if players_color == color:
                    cities[player_id][len(neighboring_colors)] += 1
                else:
                    cities[opponent_id][len(neighboring_colors)] += 1
        return cities
    
    
    
    
    
       
    # def _evaluate_board(self, state: GameState) -> int:
    #     player_id, opponent_id =  self._get_opponent_id(state)
    #     score = 0

    #     divercites = self.number_divercite(state)
    #     score += self.heuristic_divercite(divercites, player_id, opponent_id)

    #     possible_divercite = self.possible_divercite(state)
    #     score += self.heuristic_possible_divercite(possible_divercite, player_id, opponent_id)
        
    #     cities_with_same_ressources = self.cities_with_same_ressources(state)
    #     score += self.heuristic_cities_with_same_ressources(cities_with_same_ressources, player_id, opponent_id)

    #     score += state.scores[player_id] - state.scores[opponent_id]
        
    #     return score

    # def heuristic_divercite(self, divercites: Dict, player_id: int, opponent_id: int) -> int:
    #     count = 0
    #     count += divercites[player_id] * 100
    #     count -= divercites[opponent_id] * 100
    #     return count
    
    # def heuristic_possible_divercite(self, possible_divercite: Dict, player_id: int, opponent_id: int) -> int:
    #     count = 0
    #     count += possible_divercite[player_id][1] * 15
    #     count -= possible_divercite[opponent_id][1] * 15
    #     count += possible_divercite[player_id][2] * 30
    #     count -= possible_divercite[opponent_id][2] * 30
    #     count += possible_divercite[player_id][3] * 50
    #     count -= possible_divercite[opponent_id][3] * 50
    #     count += possible_divercite[player_id][4] * 75
    #     count -= possible_divercite[opponent_id][4] * 75
    #     return count
        
    # def heuristic_cities_with_same_ressources(self, cities_with_same_ressources: Dict, player_id: int, opponent_id: int) -> int:
    #     count = 0
    #     count += cities_with_same_ressources[player_id][0] * 10
    #     count -= cities_with_same_ressources[opponent_id][0] * 10
    #     count += cities_with_same_ressources[player_id][1] * 15
    #     count -= cities_with_same_ressources[opponent_id][1] * 15
    #     count += cities_with_same_ressources[player_id][2] * 25
    #     count -= cities_with_same_ressources[opponent_id][2] * 25
    #     count += cities_with_same_ressources[player_id][3] * 40
    #     count -= cities_with_same_ressources[opponent_id][3] * 40
    #     count += cities_with_same_ressources[player_id][4] * 60
    #     count -= cities_with_same_ressources[opponent_id][4] * 60
    #     return count
        
    # def number_divercite(self, state: GameState) -> Dict[str, int]:
    #     """
    #     Get the number of divercite for each player.

    #     Args:
    #         state (GameState): The current game state.

    #     Returns:
    #         Dict[str, int]: The number of divercite for each player.
    #     """
    #     player_id, opponent_id =  self._get_opponent_id(state)
    #     divercite = {
    #         player_id: 0,
    #         opponent_id: 0
    #     }
    #     for i in range(state.get_rep().get_dimensions()[0]):
    #         for j in range(state.get_rep().get_dimensions()[1]):
                
    #             if state.in_board((i, j)) and state.piece_type_match('C', (i, j)) and (i, j) in state.get_rep().get_env() and state.check_divercite((i, j)):
    #                 players_color = self.get_piece_type()
    #                 city_color = state.get_rep().get_env()[(i, j)].piece_type[-1]
    #                 if players_color == city_color:
    #                     divercite[player_id] += 1
    #                 else:   
    #                     divercite[opponent_id] += 1
    #     return divercite

    # def possible_divercite(self, state: GameState) -> Dict[str, Dict]:
    #     """
    #     Evaluate how many cities are close to divercite and to what degree.

    #     Args:
    #         state (GameState): The current game state.
    
    #     Returns:
    #         Dict[str, Dict]: The number of cities close to divercite for each player.
    #     """
    #     player_id, opponent_id =  self._get_opponent_id(state)
    #     cities = {
    #         player_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    #         opponent_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    #     }
        
    #     players_color = self.get_piece_type()
    #     cities_pos = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5), (7, 4)]
    #     for pos in cities_pos:
    #         if state.get_rep().get_env().get(pos) is not None and not state.check_divercite(pos):
    #             color = state.get_rep().get_env()[pos].piece_type[0]
    #             neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
    #             neighboring_colors = set()
    #             for neighbor, value in neighbors.items():
    #                 if value[0] != 'EMPTY':
    #                     if value[0] not in neighboring_colors:
    #                         neighboring_colors.add(value[0])
    #                     else:
    #                         continue
    #             if players_color == color:
    #                 cities[player_id][len(neighboring_colors)] += 1
    #             else:
    #                 cities[opponent_id][len(neighboring_colors)] += 1
    #     return cities
    
    # def cities_with_same_ressources(self, state: GameState) -> Dict[str, int]:
    #     """
    #     Evaluate how many cities have the same ressources.

    #     Args:
    #         state (GameState): The current game state.
    
    #     Returns:
    #         Dict[str, int]: The number of cities with the same ressources for each player.
    #     """
    #     player_id, opponent_id =  self._get_opponent_id(state)
    #     cities = {
    #         player_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    #         opponent_id: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    #     }
    #     players_color = self.get_piece_type()
    #     cities_pos = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5), (7, 4)]
    #     for pos in cities_pos:

    #         if state.get_rep().get_env().get(pos) is not None:
    #             city_color = state.get_rep().get_env()[pos].piece_type[0]
    #             neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
    #             count = 0
    #             for _, value in neighbors.items():
    #                 if value[0] != 'EMPTY' and value[0] == city_color:
    #                     count += 1
    #             if players_color == city_color:
    #                 cities[player_id][count] += 1
    #             else:
    #                 cities[opponent_id][count] += 1
    #     return cities
                    