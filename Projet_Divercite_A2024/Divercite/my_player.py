from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.game_layout.board import Piece
from seahorse.game.heavy_action import HeavyAction
from seahorse.game.light_action import LightAction
import heapq
from board_divercite import BoardDivercite
from game_state_divercite import GameStateDivercite
from typing import Dict
import copy
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
        self._max_depth = 4
        self.explored_nodes = 0
        self._seen_transposition_states = 0

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        _, best_action = self.max_value(current_state, float('-inf'), float('inf'), self._max_depth)
        print(f"Explored nodes: {self.explored_nodes}")
        print(f"Seen states: {self._seen_transposition_states}")
        return best_action


    def max_value(self, state: GameState, alpha: int, beta: int, depth: int):
        self.explored_nodes += 1
        if state in self._seen_transposition_tables:
            if self._seen_transposition_tables[state]['depth'] >= depth:
                self._seen_transposition_states += 1
                return self._seen_transposition_tables[state]['score'], self._seen_transposition_tables[state]['action']
        if state.is_done() or not depth:
            player_id, other_player_id = self.get_player_ids(state)
            return state.scores[player_id] - state.scores[other_player_id], None     
        score = float('-inf')
        action = None
        
        actions = self._generate_heap_light_action(state, isMaxHeap=True)
        while len(actions) > 0:
            a = heapq.heappop(actions)[2]
            next_state = a.get_heavy_action(state).get_next_game_state()
            value, _ = self.min_value(next_state, alpha, beta, depth - 1)             
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
        self.explored_nodes += 1
        if state in self._seen_transposition_tables:
            if self._seen_transposition_tables[state]['depth'] >= depth:
                self._seen_transposition_states += 1
                return self._seen_transposition_tables[state]['score'], self._seen_transposition_tables[state]['action']

        if state.is_done() or not depth:
            player_id, other_player_id = self.get_player_ids(state)
            return state.scores[player_id] - state.scores[other_player_id], None     
        score = float('inf')
        action = None
        actions = self._generate_heap_light_action(state, isMaxHeap=False)
        while len(actions) > 0:
            a = heapq.heappop(actions)[2]
            next_state = a.get_heavy_action(state).get_next_game_state()
            value, _ = self.max_value(next_state, alpha, beta, depth - 1)

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

    def score_delta(self, current_scores: Dict[int, float], next_scores: Dict[int, float]) -> int:
        other_player_id = [player_id for player_id in current_scores.keys() if player_id != self.get_id()][0]
        player_delta = next_scores[self.get_id()] - current_scores[self.get_id()]
        other_player_delta = next_scores[other_player_id] - current_scores[other_player_id]
        return int(player_delta - other_player_delta)

    def _generate_heap_heavy_actions(self, state: GameState, isMaxHeap: bool) -> list:
        heap = []
        heapq.heapify(heap)
        current_rep = state.get_rep()
        b = current_rep.get_env()
        d = current_rep.get_dimensions()
        
        multiplicator = -1 if isMaxHeap else 1

        for piece, n_piece in state.players_pieces_left[state.next_player.get_id()].items():
            piece_color = piece[0]
            piece_res_city = piece[1]
            if n_piece > 0:
                for i in range(d[0]):
                    for j in range(d[1]):
                        if state.in_board((i, j)) and (i,j) not in b and state.piece_type_match(piece_res_city, (i, j)):
                            copy_b = copy.copy(b)
                            copy_b[(i, j)] = Piece(piece_type=piece_color+piece_res_city+state.next_player.piece_type, owner=state.next_player)
                            play_info = ((i,j), piece, state.next_player.get_id())
                            next_state = GameStateDivercite(
                                state.compute_scores(play_info),
                                state.compute_next_player(),
                                state.players,
                                BoardDivercite(env=copy_b, dim=d),
                                step=state.step + 1,
                                players_pieces_left=state.compute_players_pieces_left(play_info),
                            )
                            score_delta = self.score_delta(state.scores, next_state.scores)
                            heapq.heappush(heap, (multiplicator * score_delta, id(HeavyAction(state, next_state)), HeavyAction(state, next_state)))
        return heap

    def _generate_heap_light_action(self, state: GameState, isMaxHeap: bool) -> list:
        heap = []
        current_rep = state.get_rep()
        b = current_rep.get_env()
        d = current_rep.get_dimensions()
        free_positions = self._get_free_positions(b, d)
        multiplicator = -1 if isMaxHeap else 1
        for piece, n_piece in state.players_pieces_left[state.next_player.get_id()].items():
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