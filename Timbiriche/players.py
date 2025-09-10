import random
import time

class MinimaxPlayer:
    """Jugador que decide usando Minimax."""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.nodes_explored = 0

    def minimax(self, state, depth, maximizing):
        """Implementación recursiva de Minimax."""
        self.nodes_explored += 1
        if depth == 0 or state.is_game_over():
            return state.evaluate_position()

        moves = state.get_possible_moves()
        if maximizing:
            value = float("-inf")
            for m in moves:
                new_state = state.copy_game_state()
                new_state.make_move(*m)
                value = max(value, self.minimax(new_state, depth-1, new_state.current_player==0))
            return value
        else:
            value = float("inf")
            for m in moves:
                new_state = state.copy_game_state()
                new_state.make_move(*m)
                value = min(value, self.minimax(new_state, depth-1, new_state.current_player==0))
            return value

    def get_best_move(self, state):
        """Selecciona el mejor movimiento evaluando con Minimax."""
        moves = state.get_possible_moves()
        if not moves:
            return None
        best = None
        best_val = float("-inf") if state.current_player == 0 else float("inf")

        for m in moves:
            new_state = state.copy_game_state()
            new_state.make_move(*m)
            val = self.minimax(new_state, self.max_depth-1, new_state.current_player==0)
            if state.current_player == 0 and val > best_val:
                best_val, best = val, m
            elif state.current_player == 1 and val < best_val:
                best_val, best = val, m
        return best


class RandomPlayer:
    """Jugador que mueve al azar."""
    def get_best_move(self, state):
        moves = state.get_possible_moves()
        return random.choice(moves) if moves else None
