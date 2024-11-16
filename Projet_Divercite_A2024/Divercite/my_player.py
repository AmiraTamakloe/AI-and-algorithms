from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
import heapq
from board_divercite import BoardDivercite
from typing import Dict
import time
import json

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "main_player"):
        """
        Initialize the PlayerDivercite instance.
        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        self._seen_transposition_tables = {}
        self.number_of_remaining_moves = 20
        self._timeout = 0
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

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.
        Uti

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        action = None
        self._timeout = time.time() + (self._time_allocation[ self.number_of_remaining_moves] * 60)- 0.5
        max_depth = 0
        while time.time() < self._timeout and self.number_of_remaining_moves * 2 > max_depth:
            if time.time() > self._timeout:
                break
            _, action = self.max_value(current_state, float('-inf'), float('inf'), max_depth)
            max_depth += 1
        # player_id, other_player_id = self.get_player_ids(current_state)
        # with open("infos.csv", "a") as f:
        #     f.write(f"{self.number_of_remaining_moves},{max_depth},{player_id},{current_state.scores[player_id]},{self._timeout},{time.time()},{other_player_id},{current_state.scores[other_player_id]}\n")
        # f.close()
        self.number_of_remaining_moves -= 1
        return action


    def max_value(self, state: GameState, alpha: int, beta: int, current_depth: int):
        """
        Return the maximum value of the state and the action to take to get to that state.
        
        Args:
            state (GameState): The current game state.
            alpha (int): The alpha value.
            beta (int): The beta value.
            current_depth (int): The depth of the search.
        Returns:
            Tuple[int, Action]: The maximum value of the state and the action to take to get to that state.
        """
        if hash(state) in self._seen_transposition_tables:
            if self._seen_transposition_tables[hash(state)]['depth'] >= current_depth:
                return self._seen_transposition_tables[hash(state)]['score'], self._seen_transposition_tables[hash(state)]['action']
            
        if time.time() > self._timeout or state.is_done() or not current_depth:
            player_id, other_player_id = self.get_player_ids(state)
            return state.scores[player_id] - state.scores[other_player_id], None     
        score = float('-inf')
        action = None
        actions = self._generate_heap_light_action(state, is_max_heap=True)
        while len(actions) > 0:
            a = heapq.heappop(actions)[2]
            if a is None:
                # FIXME Bug where the action is None
                return next(state.generate_possible_heavy_actions()), score
            next_state = a.get_heavy_action(state).get_next_game_state()
            value, _ = self.min_value(next_state, alpha, beta, current_depth - 1)
            if value > score:
                score = value
                action = a
                alpha = max(alpha, score)

            if score >= beta:
                self._seen_transposition_tables[hash(state)] = {'score': score, 'action': action, 'depth': current_depth}
                return score, action
            if time.time() > self._timeout:
                break
        self._seen_transposition_tables[hash(state)] = {'score': score, 'action': action, 'depth': current_depth}
        return score, action

    def min_value(self, state: GameState, alpha: int, beta: int, current_depth: int):
        """
        Return the minimum value of the state and the action to take to get to that state.

        Args:
            state (GameState): The current game state.
            alpha (int): The alpha value.
            beta (int): The beta value.
            current_depth (int): The depth of the search.
        Returns:
            Tuple[int, Action]: The minimum value of the state and the action to take to get to that state.
        """
        if hash(state) in self._seen_transposition_tables:
            if self._seen_transposition_tables[hash(state)]['depth'] >= current_depth:
                return self._seen_transposition_tables[hash(state)]['score'], self._seen_transposition_tables[hash(state)]['action']

        if time.time() > self._timeout or state.is_done() or not current_depth:
            player_id, other_player_id = self.get_player_ids(state)
            return state.scores[player_id] - state.scores[other_player_id], None
        score = float('inf')
        action = None
        actions = self._generate_heap_light_action(state, is_max_heap=False)
        while len(actions) > 0:
            a = heapq.heappop(actions)[2]
            if a is None:
                # FIXME Bug where the action is None
                return next(state.generate_possible_heavy_actions()), score
            next_state = a.get_heavy_action(state).get_next_game_state()
            value, _ = self.max_value(next_state, alpha, beta, current_depth - 1)

            if value < score:
                score = value
                action = a
                beta = min(beta, score)

            if score <= alpha or time.time() > self._timeout:
                self._seen_transposition_tables[hash(state)] = {'score': score, 'action': action, 'depth': current_depth}
                return score, action        
            if time.time() > self._timeout:
                break
        self._seen_transposition_tables[hash(state)] = {'score': score, 'action': action, 'depth': current_depth}
        return score, action

    def get_player_ids(self, state):
        """
        Get the player ids of our player and the other player.

        Args:
            state (GameState): The current game state.
        Returns:
            Tuple[int, int]: The player ids of our player and the other player.
        """
        keys = list(state.scores.keys())
        return (keys[0], keys[1]) if keys[0] == self.get_id() else (keys[1], keys[0])

    def score_delta(self, current_scores: Dict[int, float], next_scores: Dict[int, float]) -> int:
        """
        Calculate the difference in score between the current and next state.

        Args:
            current_scores (Dict[int, float]): The current score of each player.
            next_scores (Dict[int, float]): The next score of each player.
        Returns:
            int: The difference in score between the current and next state.
        """
        other_player_id = [player_id for player_id in current_scores.keys() if player_id != self.get_id()][0]
        player_delta = next_scores[self.get_id()] - current_scores[self.get_id()]
        other_player_delta = next_scores[other_player_id] - current_scores[other_player_id]
        return int(player_delta - other_player_delta)

    def prioritize_divercite(self, current_scores: Dict[int, float], next_scores: Dict[int, float]) -> int:
        """
        Heuristic that will put emphasis on moves where our player can do divercite and where the opponent cannot.

        Args:
            current_scores (Dict[int, float]): The current score of each player.
            next_scores (Dict[int, float]): The next score of each player.
        Returns:
            heuristic_score: The value of the move.
        """
        heuristic_score = 0
        other_player_id = [player_id for player_id in current_scores.keys() if player_id != self.get_id()][0]
        if next_scores[self.get_id()] - current_scores[self.get_id()] == 5:
            heuristic_score += 10
        elif next_scores[other_player_id] - current_scores[other_player_id] == 5:
            heuristic_score -= 10
        else:
            return self.score_delta(current_scores=current_scores, next_scores=next_scores)
        return heuristic_score
    
    def combination(self, current_scores: Dict[int, float], next_scores: Dict[int, float]):
        """
        Heuristic that will put emphasis on moves where our player can win or do divercite and where the opponent cannot.

        Args:
            current_scores (Dict[int, float]): The current score of each player.
            next_scores (Dict[int, float]): The next score of each player.
        Returns:
            heuristic_score: The value of the move.
        """
        heuristic_score = 0
        other_player_id = [player_id for player_id in current_scores.keys() if player_id != self.get_id()][0]
        if next_scores[self.get_id()] == 40:
            heuristic_score += 20
        elif next_scores[other_player_id] == 40:
            heuristic_score -= 20
        return heuristic_score + self.prioritize_divercite(current_scores=current_scores, next_scores=next_scores)

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
                        score_delta = self.score_delta(state.scores, state.compute_scores(((i,j), piece, state.next_player.get_id())))
                        action = LightAction(data)
                        heapq.heappush(heap, (multiplicator * score_delta, id(action), action))
        return heap

    def _get_free_positions(self, b, d):
        board = BoardDivercite(env=b, dim=d)
        grid_data = board.get_grid()
        return [(i, j) for i, row in enumerate(grid_data) for j, cell in enumerate(row) if (cell == ('◇ ', 'Black') or cell == ('▢ ', 'Black'))]