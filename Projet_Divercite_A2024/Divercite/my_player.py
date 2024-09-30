from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

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
        self.seen_boards = []
        self.max_depth = 3

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        _, best_action = self.max_value(current_state, float('-inf'), float('inf'), 0)
        return best_action


    def max_value(self, state: GameState, alpha: int, beta: int, depth: int):
        # if terminal return
        if depth >= self.max_depth:
            return state.scores[self.get_id()], None
        
        score = float('-inf')
        action = None
        # generate_possible_heavy_actions sera changée par fonction prenant en considération les heuristiques
        for a in state.generate_possible_heavy_actions():
            next_state = a.get_next_game_state()
            value, _ = self.min_value(next_state, alpha, beta, depth + 1)
            if value > score:
                score = value
                action = a
                alpha = max(alpha, score)

            if score >= beta:
                return score, action
            
        return score, action

    def min_value(self, state: GameState, alpha: int, beta: int, depth: int):
        if depth >= self.max_depth:
            return state.scores[self.get_id()], None
        
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
                return score, action
            
        return score, action