import copy

class DotsAndBoxes:
    """
    Implementación del juego Dots and Boxes.
    Administra el tablero, reglas, puntuación y estado del juego.
    """
    def __init__(self, size=5):
        """
        Inicializa el tablero.
        
        Args:
            size (int): Tamaño del tablero (default 5x5).
        """
        self.size = size
        self.horizontal_lines = [[False for _ in range(size)] for _ in range(size + 1)]
        self.vertical_lines = [[False for _ in range(size + 1)] for _ in range(size)]
        self.completed_boxes = [[False for _ in range(size)] for _ in range(size)]
        self.scores = [0, 0]      # [MAX, MIN]
        self.current_player = 0   # 0=MAX, 1=MIN
        self.move_history = []
        self.total_moves = 0

    # -----------------------------
    # Métodos de estado del tablero
    # -----------------------------
    def get_possible_moves(self):
        """Retorna todas las líneas disponibles como movimientos válidos."""
        moves = []
        for r, row in enumerate(self.horizontal_lines):
            for c, line in enumerate(row):
                if not line:
                    moves.append(('H', r, c))
        for r, row in enumerate(self.vertical_lines):
            for c, line in enumerate(row):
                if not line:
                    moves.append(('V', r, c))
        return moves

    def is_game_over(self):
        """Indica si ya no hay movimientos disponibles."""
        return not any(False in row for row in self.horizontal_lines + self.vertical_lines)

    def copy_game_state(self):
        """Devuelve una copia profunda del estado actual del juego."""
        new_game = DotsAndBoxes(self.size)
        new_game.horizontal_lines = copy.deepcopy(self.horizontal_lines)
        new_game.vertical_lines = copy.deepcopy(self.vertical_lines)
        new_game.completed_boxes = copy.deepcopy(self.completed_boxes)
        new_game.scores = self.scores.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.total_moves = self.total_moves
        return new_game

    # -----------------------------
    # Reglas del juego
    # -----------------------------
    def check_completed_box(self, r, c):
        """Verifica si un cuadro específico está cerrado en sus 4 lados."""
        return (self.horizontal_lines[r][c] and
                self.horizontal_lines[r+1][c] and
                self.vertical_lines[r][c] and
                self.vertical_lines[r][c+1])

    def count_box_sides(self, r, c):
        """Cuenta cuántos lados tiene dibujados un cuadro."""
        sides = 0
        if self.horizontal_lines[r][c]: sides += 1
        if self.horizontal_lines[r+1][c]: sides += 1
        if self.vertical_lines[r][c]: sides += 1
        if self.vertical_lines[r][c+1]: sides += 1
        return sides

    def make_move(self, move_type, r, c):
        """
        Realiza un movimiento y actualiza el tablero.
        
        Returns:
            int: cantidad de cuadros completados con este movimiento.
        """
        if move_type == 'H':
            if self.horizontal_lines[r][c]: return 0
            self.horizontal_lines[r][c] = True
        elif move_type == 'V':
            if self.vertical_lines[r][c]: return 0
            self.vertical_lines[r][c] = True

        self.move_history.append((move_type, r, c, self.current_player))
        self.total_moves += 1

        completed = 0
        boxes = []
        if move_type == 'H':
            if r > 0: boxes.append((r-1, c))
            if r < self.size: boxes.append((r, c))
        else:
            if c > 0: boxes.append((r, c-1))
            if c < self.size: boxes.append((r, c))

        for br, bc in boxes:
            if not self.completed_boxes[br][bc] and self.check_completed_box(br, bc):
                self.completed_boxes[br][bc] = True
                self.scores[self.current_player] += 1
                completed += 1

        if completed == 0:
            self.current_player = 1 - self.current_player

        return completed

    # -----------------------------
    # Evaluación heurística
    # -----------------------------
    def evaluate_position(self):
        """Evalúa el estado del tablero con una heurística."""
        if self.is_game_over():
            return (self.scores[0] - self.scores[1]) * 100

        # Diferencia de puntos
        score = (self.scores[0] - self.scores[1]) * 50
        
        # Cuadros casi completados
        for r in range(self.size):
            for c in range(self.size):
                if not self.completed_boxes[r][c]:
                    sides = self.count_box_sides(r, c)
                    if sides == 3:
                        score += 20 if self.current_player == 0 else -20
                    elif sides == 2:
                        score += 6 if self.current_player == 0 else -6

        # Movimientos disponibles
        score += len(self.get_possible_moves()) * 2

        # Control de turno
        if self.move_history and self.move_history[-1][3] == self.current_player:
            score += 3 if self.current_player == 0 else -3
        return score
