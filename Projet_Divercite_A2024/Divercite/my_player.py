from collections import defaultdict
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
import time
import heapq
from seahorse.game.light_action import LightAction
from board_divercite import BoardDivercite
from typing import Dict

MAX_DEPTH = 40
TWO_PLY_STEP = 2
CITIES_POS = [(1, 4), (2, 3), (2, 5), (3, 2), (3, 4), (3, 6), (4, 1), (4, 3), (4, 5), (4, 7), (5, 2), (5, 4), (5, 6), (6, 3), (6, 5), (7, 4)]

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
            20:0.2, 19:0.2, 18:0.3, 17:0.4, 16:0.6, 15:0.7, 14:1.1, 13:1.2, 12:1.3, 11:1.6, 
            10:1.55, 9:1.45, 8:1.45, 7:1.45, 6:1.2, 5:0.1, 4:0.05, 3:0.05, 2:0.05, 1:0.05
          }
        self.phase = 'EARLY'

    def _handle_phase_change(self, state: GameState):
        """
        Update the game phase based on the number of remaining moves.

        Args:
            state (GameState): The current game state.
        """

        _number_of_remaining_moves = sum(state.players_pieces_left[self.get_id()].values())
        if 16 < _number_of_remaining_moves:
            self.phase = 'EARLY'
        elif 7 < _number_of_remaining_moves :
            self.phase = 'MID'
        else:
            self.phase = 'LATE'

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Compute the best action using iterative deepening minimax with alpha-beta pruning.

        This function determines the best action for the AI player by employing an 
        iterative deepening approach to the minimax algorithm with alpha-beta pruning. 
        It progressively explores the search tree to increasing depths, allowing it 
        to make an informed decision even if the search is interrupted due to time 
        constraints.

        The search depth increases in steps of `TWO_PLY_STEP` until the maximum depth 
        (`MAX_DEPTH`) is reached or the allocated time runs out. This ensures that 
        the algorithm returns the best action found within the available time.

        Args:
            current_state (GameState): The current game state.
            remaining_time (int): The remaining time for the AI to make a move (not used).

        Returns:
            Action: The best action determined by the minimax search.
        """
        self._handle_phase_change(current_state)

        action = None
        self._timeout = time.time() + (self._time_allocation[ self._number_of_remaining_moves] * 60)- 0.5

        for depth in range(1, MAX_DEPTH + 1, TWO_PLY_STEP):
            if time.time() > self._timeout:
                break

            _, action = self._minimax(current_state, depth, float('-inf'), float('inf'), maximizing=True)
        self._number_of_remaining_moves -= 1

        return action
    
    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing):
        """
        Minimax algorithm with alpha-beta pruning and transposition tables.

        This function implements a recursive depth-limited minimax search with 
        alpha-beta pruning to determine the best action for the current player 
        in the given game state. It uses a transposition table to store previously 
        evaluated states, improving search efficiency by avoiding redundant calculations.

        Args:
            state (GameState): The current game state.
            depth (int): The remaining search depth.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.
            maximizing (bool): True if the current player is maximizing, False otherwise.

        Returns:
            Tuple[int, Action]: A tuple containing:
                                - The best score achievable from this state.
                                - The best action to take to achieve that score.
        """

        self._handle_phase_change(state)
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
                bestMove = heapq.heappop(actions)[2]
                minEval = self._evaluate_board(bestMove.get_heavy_action(state).get_next_game_state())
            
            self._transposition_table[hash(state)] = {'depth': depth, 'score': minEval}
            return minEval, bestMove

    def _generate_heap_light_action(self, state: GameState, is_max_heap: bool) -> list:
        """
        Generate a heap of possible actions prioritized by their heuristic score.

        This function creates a heap data structure containing all possible 
        `LightAction`s for the current player in the given game state. Each action 
        is associated with a score calculated by the `_evaluate_board` heuristic. 

        The heap is sorted based on these scores, allowing efficient retrieval of 
        the highest-priority (for maximizing player) or lowest-priority (for 
        minimizing player) actions. This prioritization helps guide the search 
        algorithm towards more promising game states.

        Args:
            state (GameState): The current game state.
            is_max_heap (bool):  If True, the heap is sorted for a maximizing player 
                                (highest scores first). If False, it's sorted for a 
                                minimizing player (lowest scores first).

        Returns:   
            list: A heap of tuples, where each tuple contains:
                - The action's score (multiplied by -1 for max_heap)
                - The action's ID (for tie-breaking)
                - The `LightAction` object
        """

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

    def _evaluate_board(self, state: GameState) -> int:
        """
        Evaluate the game state based on the current phase of the game.

        Args:
            state (GameState): The current game state.

        Returns:
            int: The evaluation score. Higher score indicates a better game state.
        """
    
        if self.phase == 'EARLY':
            return self._evaluate_early(state)
        elif self.phase == 'MID':
            return self._evaluate_mid(state)
        else:
            return self._evaluate_late(state)

    def _get_opponent_id(self, state: GameState):
        """
        Get the opponent's player ID.

        Args:
            state (GameState): The current game state.

        Returns:
            str: The opponent's player ID.
        """

        keys = list(state.scores.keys())
        return keys[1] if keys[0] == self.get_id() else keys[0]
    
    def _get_free_positions(self, b, d):
        """
        Get the free positions on the board.

        Args:
            b (Dict): The board data.
            d (Tuple): The board dimensions.

        Returns:
            List[Tuple[int, int]]: A list of free positions on the board.
        """

        board = BoardDivercite(env=b, dim=d)
        grid_data = board.get_grid()
        return [(i, j) for i, row in enumerate(grid_data) for j, cell in enumerate(row) if (cell == ('◇ ', 'Black') or cell == ('▢ ', 'Black'))]
    
    def _get_score_difference(self, state: GameState) -> int:
        """
        Calculate the difference between the player's score and the opponent's score.

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score difference. A positive value indicates the player is leading.
        """

        id_opponent = self._get_opponent_id(state)
        return state.scores[self.get_id()] - state.scores[id_opponent]

    def _evaluate_early(self, state: GameState) -> int:
        """
        Evaluate the game state for the early-game phase.

        This heuristic combines several factors to assess the game state during the early-game:

        1. Variety of Resources and Cities:
        - Rewards having a variety of resource types and city colors used (`_verify_variety`).

        2. Placement:
        - Rewards having cities placed in central positions (`_placement`).

        3. Score Evaluation:
        - Rewards having a higher score compared to the opponent (`_evaluate`).
        """

        score = 0
        score += self._verify_variety(state)
        score += self._placement(state)
        score += self._get_score_difference(state)
        return score
    
    def _evaluate_mid(self, state: GameState) -> int:
        """
        Evaluate the game state for the mid-game phase.

        This heuristic combines several factors to assess the game state during the mid-game:

        1. Cities close to do divercite: 
        - Rewards having cities that could soon do a divercite (`_closeness_divercite`).

        2. Avoiding Blocking Cities:
        - Penalizes having cities that cannot do a divercite by having multiple time the same color around it (`_blocked_divercite`).
        - This promotes strategic placement to maintain access to resources.

        3. Score Evaluation:
        - Rewards having a higher score compared to the opponent (`_evaluate`).

        Args:
            state (GameState): The current game state.

        Returns:
            int: The evaluation score. Higher score indicates a better game state.
        """

        score = 0
        score += self._closeness_divercite(state)
        score -= self._blocked_divercite(state)
        score += self._get_score_difference(state)
        return score

    def _evaluate_late(self, state: GameState) -> int:
        """
        Evaluate the game state for the late game phase.

        This heuristic prioritizes winning above all else. If the game is finished, 
        it returns a decisive score based on the outcome: a high score for a win 
        and a low score for a loss.

        If the game is still ongoing, it falls back to a score evaluation function 
        (_evaluate).

        Args:
            state (GameState): The current game state.

        Returns:
            int: The evaluation score.
        """

        if state.is_done():
            scores = state.remove_draw(state.scores, state.get_rep())
            if scores[self.get_id()] > scores[self._get_opponent_id(state)]:
                return 10000
            else:
                return -10000
        else:
            return self._get_score_difference(state)

    def _verify_variety(self, state: GameState) -> int:
        """
        Evaluate the balance and variety of resources and cities used by the player.

        This heuristic encourages a balanced strategy by considering two factors:

        1. Variety of Resources and Cities:
        - Calculates separate scores for the variety of resource types and city colors used.
        - Higher variety leads to a higher score.

        2. Resource-City Ratio:
        - Checks if the player is using resources and cities at a roughly similar pace.
        - Penalizes excessive resource accumulation without corresponding city placement.

        Args:
            state (GameState): The current game state.

        Returns:
            int: The balance and variety score. Higher score indicates better balance and variety.
        """

        score = 0
        items = state.players_pieces_left[self.get_id()]

        filter_items = lambda item_type: {item: quantity for item, quantity in items.items() if item[1] == item_type}

        resources = defaultdict(int, filter_items('R'))
        cities = defaultdict(int, filter_items('C')) 

        # Check if are using too much of one color
        variety_resources = self._variety_resources(resources) 
        variety_cities = self._variety_cities(cities)
        score += (variety_resources + variety_cities) 

        # Check if we are using more resources than cities
        used_resources = (12 - sum(resources.values())) / 12
        used_cities = (8 - sum(cities.values())) / 8
        ratio = (used_cities) / max(used_resources, 1)
        if ratio > 0.5 and ratio < 2:
            score += 2
        else:
            score -= 2

        return score
    
    def _variety_resources(self, resources: Dict) -> int:
        """
        Calculate a score representing the variety of resources used by the player.

        The score is determined by the difference between the number of times the 
        most used resource and the least used resource are used. A lower difference 
        indicates a higher variety, resulting in a higher score.

        Args:
            resources (Dict): A dictionary where keys represent resource types and 
                               values represent the count of each resource type.

        Returns:
            int: A score representing the variety of resources used. 
                 Higher score indicates more variety.
        """

        resources_count = [value for value in resources.values()]
        min_count = min(resources_count)
        max_count = max(resources_count)
        return 3 - (max_count - min_count)
    
    def _variety_cities(self, resources: Dict) -> int:
        """
        Calculate a score representing the variety of cities used by the player.

        The score is determined by the difference between the number of times the 
        most used city and the least used city are used. A lower difference 
        indicates a higher variety, resulting in a higher score.

        Args:
            resources (Dict): A dictionary where keys represent city types and 
                               values represent the count of each city type.

        Returns:
            int: A score representing the variety of cities used. 
                 Higher score indicates more variety.
        """

        cities_count = [value for value in resources.values()]
        min_count = min(cities_count)
        max_count = max(cities_count)
        return 2 - (max_count - min_count)
    
    def _placement(self, state: GameState) -> int:
        """
        Evaluate the placement of the player's cities, prioritizing central positions.

        This heuristic assigns a higher score to games states where the player has 
        more cities placed in the center of the board. The center is defined as 
        the four tiles with coordinates (3, 4), (4, 3), (4, 5), and (5, 4).

        Args:
            state (GameState): The current game state.

        Returns:
            int: The placement score. Higher score indicates better placement.
        """

        score = 0
        CENTRAL_CITIES = [(3, 4), (4, 3), (4, 5), (5, 4)]
        for city in CENTRAL_CITIES:
            if state.get_rep().get_env().get(city) is not None:
                color = state.get_rep().get_env()[city].piece_type[-1]
                if color == self.get_piece_type():
                    score += 1
        return score
    
    def _closeness_divercite(self, state: GameState) -> int:
        """
        Evaluate how many cities are close to make a divercite for each player.

        This heuristic assigns a higher score to game states where the player has 3 resources of different colors around his cities
        and the correct resource to do a divercite.

        Args:
            state (GameState): The current game state.

        Returns:
            int: The score of the evaluation. It will be positive if the opponant has more possible divercites than the player.
        """

        possible_divercite = self.possible_divercite(state)
        opponent_id = self._get_opponent_id(state)
        score = 0
        score -= possible_divercite[opponent_id]
        score += possible_divercite[self.get_id()]
        return score

    def possible_divercite(self, state: GameState) -> Dict[str, int]:
        """
        Evaluate how many cities are close to make a divercite.

        Args:
            state (GameState): The current game state.
    
        Returns:
            Dict[str, int]: The number of cities that could do divercite for each player.
        """

        opponent_id =  self._get_opponent_id(state)
        cities = {
            self.get_id(): 0,
            opponent_id: 0,
        } 
        players_color = self.get_piece_type()
        remaining_color = {
            colors[0] 
            for colors, value in state.players_pieces_left[self.get_id()].items() 
            if colors[1] == 'R' and value != 0 
        }
        opponent_remaining_color = {
            colors[0] 
            for colors, value in state.players_pieces_left[opponent_id].items() 
            if colors[1] == 'R' and value != 0 
        }

        comparaison_set = set(['B', 'Y', 'R', 'G'])
        skip_position = False
        for pos in CITIES_POS:
            if state.get_rep().get_env().get(pos) is not None and not state.check_divercite(pos):
                color = state.get_rep().get_env()[pos].piece_type[-1]
                neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
                neighboring_colors = set()
                for _, value in neighbors.items():
                    if value[0] != 'EMPTY':
                        if value[0].piece_type[0] not in neighboring_colors:
                            neighboring_colors.add(value[0].piece_type[0])
                        else:
                            skip_position = True
                            break 

                if skip_position:
                    skip_position = False
                    continue
       
                if len(neighboring_colors) > 2:
                    added_value = 5 if state.get_rep().get_env()[pos].piece_type[0] in neighboring_colors else 4
                    if players_color == color and remaining_color | neighboring_colors == comparaison_set:
                        cities[self.get_id()] += added_value
                    elif opponent_remaining_color | neighboring_colors == comparaison_set:
                        cities[opponent_id] += added_value      
        return cities
    
    def _blocked_divercite(self, state: GameState) -> int:
        """
        Count the number of player's cities blocked from achieving diversity.

        A city is considered "blocked" if it cannot achieve resource diversity, 
        meaning it has duplicate resource types in its neighboring tiles. 
        This function iterates through potential city positions and checks if 
        a placed city at that position is blocked.

        Args:
            state (GameState): The current game state.

        Returns:
            int: The number of blocked cities.
        """

        blocked = 0
        for pos in CITIES_POS:
            if state.get_rep().get_env().get(pos) is not None and not state.check_divercite(pos):
                neighbors = state.get_rep().get_neighbours(pos[0], pos[1])
                neighboring_colors = set()
                for _, value in neighbors.items():
                    if value[0] != 'EMPTY':
                        if value[0].piece_type[0] not in neighboring_colors:
                            neighboring_colors.add(value[0].piece_type[0])
                        else:
                            blocked += 1
                            break
        return blocked