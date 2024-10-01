from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from board_divercite import BoardDivercite
import random
class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.
        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        self._seen_transposition_tables = {}
        self.max_depth = 5
        self.node_count = 0

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        _, best_action = self.max_value(current_state, float('-inf'), float('inf'), 0)
        print("Node count: ", self.node_count)
        return best_action


    def max_value(self, state: GameState, alpha: int, beta: int, depth: int):
        self.node_count += 1
        if state in self._seen_transposition_tables:
            if self._seen_transposition_tables[state]['depth'] >= depth:
                return self._seen_transposition_tables[state]['score'], self._seen_transposition_tables[state]['action']
                    

        if state.is_done() or depth >= self.max_depth:
            player_id, other_player_id = self.get_player_ids(state)
            return state.scores[player_id] - state.scores[other_player_id], None

        
        score = float('-inf')
        action = None
        for a in state.generate_possible_heavy_actions():
            next_state = a.get_next_game_state()
            value, _ = self.min_value(next_state, alpha, beta, depth + 1)
                
            if value > score:
                score = value
                action = a
                alpha = max(alpha, score)

            if score >= beta:
                self.record_seen_board(state, score, action, depth)
                return score, action
    
        self.record_seen_board(state, score, action, depth)
        return score, action

    def min_value(self, state: GameState, alpha: int, beta: int, depth: int):
        self.node_count += 1
        if state in self._seen_transposition_tables:
            if self._seen_transposition_tables[state]['depth'] >= depth:
                    return self._seen_transposition_tables[state]['score'], self._seen_transposition_tables[state]['action']

        if state.is_done() or depth >= self.max_depth:
            player_id, other_player_id = self.get_player_ids(state)
            return state.scores[player_id] - state.scores[other_player_id], None
        
        score = float('inf')
        action = None

        for a in state.generate_possible_heavy_actions():
            next_state = a.get_next_game_state()
            
            value, _ = self.max_value(next_state, alpha, beta, depth + 1)

            if value < score:
                score = value
                action = a
                beta = min(beta, score)

            if score <= alpha:
                self.record_seen_board(state, score, action, depth)
                return score, action
            
        
        self.record_seen_board(state, score, action, depth)
        return score, action


    def record_seen_board(self, hash, score, action, depth):
        self._seen_transposition_tables[hash] = {}
        self._seen_transposition_tables[hash]['score'] = score
        self._seen_transposition_tables[hash]['action'] = action
        self._seen_transposition_tables[hash]['depth'] = depth

    def get_player_ids(self, state):
        keys = list(state.scores.keys())
        return (keys[0], keys[1]) if keys[0] == self.get_id() else (keys[1], keys[0])
