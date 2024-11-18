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
        free_positions = len(self._get_free_positions(state.get_rep().get_env(), state.get_rep().get_dimensions()))
        player_pieces_left = sum(state.players_pieces_left[self.get_id()].values())
        opponent_pieces_left = sum(state.players_pieces_left[self._get_opponent_id(state)[1]].values())
        
        if free_positions > 50 and player_pieces_left > 10 and opponent_pieces_left > 10:
            self.phase = 'EARLY'
        elif 20 <= free_positions <= 50:
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
        if hash(state) in self._transposition_table:
            stored_depth, stored_score = self._transposition_table[hash(state)]['depth'], self._transposition_table[hash(state)]['score']
            if stored_depth >= depth:
                return stored_score, None

        if depth == 0 or state.is_done():
            score = self._evaluate_board(state)
            self._transposition_table[hash(state)] = {'depth': depth, 'score': score}
            return score, None
        
        bestMove = None
        if maximizing:
            maxEval = float('-inf')
            actions = self._generate_heap_light_action(state, is_max_heap=True)
            while len(actions) > 0 and time.time() > self._timeout:
                action = heapq.heappop(actions)[2]
                score, _ = self._minimax(action.get_heavy_action(state).get_next_game_state(), depth - 1, alpha, beta, False)
                if score > maxEval:
                    maxEval = score
                    bestMove = action
        
                alpha = max(alpha, maxEval)
                if beta <= alpha:
                    break # pruning

            self._transposition_table[hash(state)] = {'depth': depth, 'score': maxEval}    
            return maxEval, bestMove
        
        else:
            minEval = float('inf')
            actions = self._generate_heap_light_action(state, is_max_heap=False)
            while len(actions) > 0 and time.time() > self._timeout:
                action = heapq.heappop(actions)[2]
                score, _ = self._minimax(action.get_heavy_action(state).get_next_game_state(), depth - 1, alpha, beta, True)
                if score < minEval:
                    minEval = score
                    bestMove = action

                beta = min(beta, minEval)
                if beta <= alpha:
                    break # pruning
    
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
            if self._timeout < time.time():
                break

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
        # Prioritize cities with spacing and close ressources opponents
            # Don't block ourself from getting divercite
        # Colour variety
        
        pass

    def _evaluate_mid(self, state: GameState) -> int:
        # Prioritize the number of divercite
        # Block other player from getting divercite
        # If no divercite, prioritize the number of cities with the same ressources
        pass

    def _evaluate_late(self, state: GameState) -> int:
        if state.is_done():
            scores = state.remove_draw(state.scores, state.get_rep())
            if scores[self.get_id()] > scores[self._get_opponent_id(state)[1]]:
                return 1000
            else:
                return -1000
        else:
            return self.evaluate(state) # TODO: add a better evaluation function

    def _verify_variety(self, state: GameState) -> int:
        
        pass
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
                    