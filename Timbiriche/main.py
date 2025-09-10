import random
import copy
import time

class DotsAndBoxes: 
    """
    Implementación del juego Dots and Boxes para un tablero 3x3.
    
    Representación del tablero:
    - horizontal_lines[fila][columna]: línea horizontal del punto (fila,columna) al (fila,columna+1)
    - vertical_lines[fila][columna]: línea vertical del punto (fila,columna) al (fila+1,columna)  
    - completed_boxes[fila][columna]: cuadro en posición (fila,columna) completado
    - scores[jugador]: puntuación de cada jugador (0=MAX, 1=MIN)
    - current_player: jugador actual (0=MAX, 1=MIN)
    """
    
    def __init__(self, size=5):
        """
        Inicializa el tablero de Dots and Boxes.
        
        Args:
            size (int): Tamaño del tablero (por defecto 3x3)
        """
        self.size = size
        
        # Matrices para representar las líneas
        # horizontal_lines: (size+1) x size
        self.horizontal_lines = [[False for _ in range(size)] for _ in range(size + 1)]
        
        # vertical_lines: size x (size+1) 
        self.vertical_lines = [[False for _ in range(size + 1)] for _ in range(size)]
        
        # completed_boxes: size x size
        self.completed_boxes = [[False for _ in range(size)] for _ in range(size)]
        
        # Puntuaciones y jugador actual
        self.scores = [0, 0]  # [MAX, MIN]
        self.current_player = 0  # 0 = MAX, 1 = MIN
        
        # Para tracking del algoritmo
        self.move_history = []
        self.total_moves = 0
    
    def print_board(self):
        """
        Imprime el estado actual del tablero en consola.
        """
        print("="*40)
        
        # Imprimir el tablero línea por línea
        for row in range(self.size + 1):
            # Imprimir puntos y líneas horizontales
            line = "•"
            for col in range(self.size):
                if row < len(self.horizontal_lines) and col < len(self.horizontal_lines[row]):
                    if self.horizontal_lines[row][col]:
                        line += "---•"
                    else:
                        line += "   •"
                else:
                    line += "   •"
            print(line)
            
            # Imprimir líneas verticales y cuadros (excepto en la última fila)
            if row < self.size:
                line = ""
                for col in range(self.size + 1):
                    if col < len(self.vertical_lines[row]):
                        if self.vertical_lines[row][col]:
                            line += "|"
                        else:
                            line += " "
                    else:
                        line += " "
                    
                    # Añadir contenido del cuadro si existe
                    if col < self.size:
                        if self.completed_boxes[row][col]:
                            line += " X "  # Marcar cuadros completados
                        else:
                            line += "   "
                print(line)
        print("="*40)
        print(f"\nPuntuación - MAX: {self.scores[0]}, MIN: {self.scores[1]}")
        print(f"Siguiente turno del jugador: {'MAX' if self.current_player == 0 else 'MIN'}")
        print("="*40)
    
    def get_possible_moves(self):
        """
        Obtiene todos los movimientos posibles (líneas no dibujadas).
        
        Returns:
            list: Lista de tuplas (tipo, fila, columna) representando movimientos válidos
                  tipo: 'H' para horizontal, 'V' para vertical
        """
        moves = []
        
        # Verificar líneas horizontales disponibles
        for row in range(len(self.horizontal_lines)):
            for col in range(len(self.horizontal_lines[row])):
                if not self.horizontal_lines[row][col]:
                    moves.append(('H', row, col))
        
        # Verificar líneas verticales disponibles  
        for row in range(len(self.vertical_lines)):
            for col in range(len(self.vertical_lines[row])):
                if not self.vertical_lines[row][col]:
                    moves.append(('V', row, col))
        
        return moves
    
    def is_game_over(self):
        """
        Verifica si el juego ha terminado (todas las líneas dibujadas).
        
        Returns:
            bool: True si el juego terminó, False en caso contrario
        """
        # Verificar si quedan líneas horizontales por dibujar
        for row in self.horizontal_lines:
            for line in row:
                if not line:
                    return False
        
        # Verificar si quedan líneas verticales por dibujar
        for row in self.vertical_lines:
            for line in row:
                if not line:
                    return False
        
        return True
    
    def check_completed_box(self, box_row, box_col):
        """
        Verifica si un cuadro específico está completado (tiene sus 4 lados dibujados).
        
        Args:
            box_row (int): Fila del cuadro
            box_col (int): Columna del cuadro
            
        Returns:
            bool: True si el cuadro está completado, False en caso contrario
        """
        # Un cuadro (box_row, box_col) está completado si tiene estos 4 lados:
        # - Lado superior: horizontal_lines[box_row][box_col]
        # - Lado inferior: horizontal_lines[box_row + 1][box_col]  
        # - Lado izquierdo: vertical_lines[box_row][box_col]
        # - Lado derecho: vertical_lines[box_row][box_col + 1]
        
        top = self.horizontal_lines[box_row][box_col]
        bottom = self.horizontal_lines[box_row + 1][box_col]
        left = self.vertical_lines[box_row][box_col]
        right = self.vertical_lines[box_row][box_col + 1]
        
        return top and bottom and left and right
    
    def make_move(self, move_type, row, col):
        """
        Realiza un movimiento dibujando una línea y actualiza el estado del juego.
        
        Args:
            move_type (str): 'H' para línea horizontal, 'V' para línea vertical
            row (int): Fila de la línea
            col (int): Columna de la línea
            
        Returns:
            int: Número de cuadros completados con este movimiento
        """
        # Verificar que el movimiento es válido
        if move_type == 'H':
            if self.horizontal_lines[row][col]:
                print(f"Error: Línea horizontal ({row},{col}) ya está dibujada")
                return 0
            self.horizontal_lines[row][col] = True
        elif move_type == 'V':
            if self.vertical_lines[row][col]:
                print(f"Error: Línea vertical ({row},{col}) ya está dibujada")
                return 0
            self.vertical_lines[row][col] = True
        else:
            print(f"Error: Tipo de movimiento inválido: {move_type}")
            return 0
        
        # Registrar el movimiento
        self.move_history.append((move_type, row, col, self.current_player))
        self.total_moves += 1
        
        # Verificar qué cuadros se completaron con este movimiento
        completed_boxes_count = 0
        boxes_to_check = []
        
        if move_type == 'H':
            # Una línea horizontal puede completar hasta 2 cuadros
            # Cuadro arriba: (row-1, col) si row > 0
            if row > 0:
                boxes_to_check.append((row - 1, col))
            # Cuadro abajo: (row, col) si row < size
            if row < self.size:
                boxes_to_check.append((row, col))
                
        elif move_type == 'V':
            # Una línea vertical puede completar hasta 2 cuadros  
            # Cuadro izquierda: (row, col-1) si col > 0
            if col > 0:
                boxes_to_check.append((row, col - 1))
            # Cuadro derecha: (row, col) si col < size
            if col < self.size:
                boxes_to_check.append((row, col))
        
        # Verificar y marcar cuadros completados
        for box_row, box_col in boxes_to_check:
            if (0 <= box_row < self.size and 0 <= box_col < self.size and
                not self.completed_boxes[box_row][box_col] and
                self.check_completed_box(box_row, box_col)):
                
                self.completed_boxes[box_row][box_col] = True
                self.scores[self.current_player] += 1
                completed_boxes_count += 1
                #print(f"¡Jugador {'MAX' if self.current_player == 0 else 'MIN'} completó cuadro ({box_row},{box_col})!")
        
        # Si no se completaron cuadros, cambiar de jugador
        if completed_boxes_count == 0:
            self.current_player = 1 - self.current_player
        
        return completed_boxes_count
    
    def copy_game_state(self):
        """
        Crea una copia profunda del estado actual del juego.
        
        Returns:
            DotsAndBoxes: Nueva instancia con el mismo estado
        """
        new_game = DotsAndBoxes(self.size)
        new_game.horizontal_lines = copy.deepcopy(self.horizontal_lines)
        new_game.vertical_lines = copy.deepcopy(self.vertical_lines)
        new_game.completed_boxes = copy.deepcopy(self.completed_boxes)
        new_game.scores = self.scores.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.total_moves = self.total_moves
        return new_game
    
    def evaluate_position(self):
        """
        Función de evaluación heurística que considera 4 factores principales.
        
        Returns:
            float: Valor de evaluación (positivo favorece a MAX, negativo a MIN)
        """
        #formas de evaluar, SUPER HEURÍSTICA:

            # Diferencia de puntuación actual = peso 50
            # Cuadros casi completos = peso 20
            # movilidad = peso 2 (cantidad de movimientos posibles)
            # Bonus por turno continuo = peso 3

        
        if self.is_game_over():
            return (self.scores[0] - self.scores[1]) * 100  # Multiplicar para dar más peso
        
        evaluation = 0.0
        
        # 1. DIFERENCIA DE PUNTUACIÓN ACTUAL (peso 50)
        score_diff = self.scores[0] - self.scores[1]
        evaluation += score_diff * 50
        
        # 2. CUADROS CASI COMPLETOS (peso 20)
        max_almost_complete = 0
        min_almost_complete = 0
        
        for row in range(self.size):
            for col in range(self.size):
                if not self.completed_boxes[row][col]:
                    sides_drawn = self.count_box_sides(row, col)
                    
                    # Cuadros con 3 lados son críticos
                    if sides_drawn == 3:
                        # El jugador actual puede completarlo = ventaja
                        if self.current_player == 0:  # MAX
                            max_almost_complete += 1
                        else:  # MIN
                            min_almost_complete += 1
                    
                    # Cuadros con 2 lados son oportunidades
                    elif sides_drawn == 2:
                        if self.current_player == 0:
                            max_almost_complete += 0.3
                        else:
                            min_almost_complete += 0.3
        
        # Favor al jugador que puede completar cuadros, penalizar al que da oportunidades
        evaluation += max_almost_complete * 20
        evaluation -= min_almost_complete * 20
        
        # 3. MOVILIDAD (peso 2) - cantidad de movimientos posibles
        mobility = len(self.get_possible_moves())
        evaluation += mobility * 2
        
        # 4. BONUS POR TURNO CONTINUO (peso 3)
        # Si el jugador actual completó cuadros en su último turno, pequeño bonus
        if len(self.move_history) > 0:
            last_move = self.move_history[-1]
            if last_move[3] == self.current_player:  # Mismo jugador que el movimiento anterior
                evaluation += 3 if self.current_player == 0 else -3
        
        return evaluation

    def count_box_sides(self, box_row, box_col):
        """
        Cuenta cuántos lados de un cuadro específico están dibujados.
        
        Args:
            box_row (int): Fila del cuadro
            box_col (int): Columna del cuadro
            
        Returns:
            int: Número de lados dibujados (0-4)
        """
        if (box_row < 0 or box_row >= self.size or 
            box_col < 0 or box_col >= self.size):
            return 0
        
        sides = 0
        
        # Lado superior
        if self.horizontal_lines[box_row][box_col]:
            sides += 1
        
        # Lado inferior
        if self.horizontal_lines[box_row + 1][box_col]:
            sides += 1
        
        # Lado izquierdo
        if self.vertical_lines[box_row][box_col]:
            sides += 1
        
        # Lado derecho  
        if self.vertical_lines[box_row][box_col + 1]:
            sides += 1
        
        return sides

class MinimaxPlayer:
    """
    Jugador que usa el algoritmo Minimax para tomar decisiones.
    """
    
    def __init__(self, max_depth):
        """
        Inicializa el jugador Minimax.
        
        """
        self.max_depth = max_depth
        self.nodes_explored = 0
        
    def minimax(self, game_state, depth, is_maximizing):
        """
        Algoritmo Minimax básico.
        
        Args:
            game_state (DotsAndBoxes): Estado actual del juego
            depth (int): Profundidad actual en el árbol
            is_maximizing (bool): True si es turno de MAX, False si es MIN
            
        Returns:
            int: Valor de evaluación del mejor movimiento
        """
        self.nodes_explored += 1
        
        # Caso base: profundidad máxima o juego terminado
        if depth == 0 or game_state.is_game_over():
            return game_state.evaluate_position()
        
        possible_moves = game_state.get_possible_moves()
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in possible_moves:
                # Simular el movimiento
                temp_game = game_state.copy_game_state()
                temp_game.make_move(move[0], move[1], move[2])
                
                # Determinar si sigue siendo el turno del mismo jugador
                # (si completó cuadros, sigue siendo su turno)
                next_is_maximizing = (temp_game.current_player == 0)
                
                eval_score = self.minimax(temp_game, depth - 1, next_is_maximizing)
                max_eval = max(max_eval, eval_score)
                
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                # Simular el movimiento
                temp_game = game_state.copy_game_state()
                temp_game.make_move(move[0], move[1], move[2])
                
                # Determinar si sigue siendo el turno del mismo jugador
                next_is_maximizing = (temp_game.current_player == 0)
                
                eval_score = self.minimax(temp_game, depth - 1, next_is_maximizing)
                min_eval = min(min_eval, eval_score)
                
            return min_eval
    
    def get_best_move(self, game_state):
        """
        Obtiene el mejor movimiento usando Minimax.
        
        Args:
            game_state (DotsAndBoxes): Estado actual del juego
            
        Returns:
            tuple: Mejor movimiento (tipo, fila, columna)
        """
        self.nodes_explored = 0
        start_time = time.time()
        
        possible_moves = game_state.get_possible_moves()
        if not possible_moves:
            return None
            
        best_move = None
        best_value = float('-inf') if game_state.current_player == 0 else float('inf')
        
        print(f"Minimax analizando {len(possible_moves)} movimientos posibles...")
        
        for i, move in enumerate(possible_moves):
            # Simular el movimiento
            temp_game = game_state.copy_game_state()
            temp_game.make_move(move[0], move[1], move[2])
            
            # Determinar el próximo jugador
            next_is_maximizing = (temp_game.current_player == 0)
            
            # Evaluar el movimiento
            move_value = self.minimax(temp_game, self.max_depth - 1, next_is_maximizing)
            
            print(f"   Movimiento {i+1}: {move} → Valor: {move_value}")
            
            # Actualizar el mejor movimiento
            if game_state.current_player == 0:  # MAX
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
            else:  # MIN
                if move_value < best_value:
                    best_value = move_value
                    best_move = move
        
        end_time = time.time()
        
        print(f"\nMejor movimiento: {best_move} (Valor: {best_value})")
        print(f"Tiempo: {end_time - start_time:.3f}s")
        print(f"Nodos explorados: {self.nodes_explored}")
        
        return best_move

class RandomPlayer:
    """
    Jugador que hace movimientos aleatorios.
    """
    
    def get_best_move(self, game_state):
        """
        Selecciona un movimiento aleatorio.
        
        """
        possible_moves = game_state.get_possible_moves()
        if not possible_moves:
            return None
        
        move = random.choice(possible_moves)
        print(f"Movimiento jugador aleatorio: {move}")
        return move

def play_game(player1, player2, sizeBoard, show_board=True):
        """
        Simula una partida entre dos jugadores.
        
        """
        game = DotsAndBoxes(sizeBoard)
        players = [player1, player2]
        
        print("¡INICIANDO PARTIDA!")
        print("Player 1 (MAX) vs Player 2 (MIN)")
        
        if show_board:
            game.print_board()
        
        while not game.is_game_over():
            current_player_obj = players[game.current_player]
            player_name = "MAX" if game.current_player == 0 else "MIN"
            print("\n" * 10)
            print(f"\n--- Turno de {player_name} ---\n")
            
            # Obtener el mejor movimiento del jugador actual
            move = current_player_obj.get_best_move(game)
            
            if move is None:
                break
                
            # Realizar el movimiento
            completed_boxes = game.make_move(move[0], move[1], move[2])
            
            if show_board:
                game.print_board()
            
            # Pausa para poder seguir el juego
            if show_board:
                input("Presiona Enter para continuar...")
        
        # Determinar ganador
        if game.scores[0] > game.scores[1]:
            winner = "MAX"
        elif game.scores[1] > game.scores[0]:
            winner = "MIN"
        else:
            winner = "EMPATE"
        
        print(f"\n RESULTADO FINAL:")
        print(f"   Ganador: {winner}")
        print(f"   Puntuación: MAX={game.scores[0]}, MIN={game.scores[1]}")
        print(f"   Total movimientos: {game.total_moves}")
        
        return winner, game.scores, game.total_moves

# Ejemplo de partida: Minimax vs Aleatorio
if __name__ == "__main__":
    print("Timbirichi- MINIMAX VS ALEATORIO")
    print("="*50)
    
    minimax_player = MinimaxPlayer(max_depth=3)  
    random_player = RandomPlayer()


    sizeBoard = 5
    
    # Jugar una partida
    winner, scores, moves = play_game(minimax_player, random_player, sizeBoard, show_board=True)
    
    print("="*50)
    print(f"\nESTADÍSTICAS FINALES:")
    print(f"   Minimax (MAX): {scores[0]} puntos")
    print(f"   Aleatorio (MIN): {scores[1]} puntos") 
    print(f"   Movimientos totales: {moves}")
    
    # Opción para jugar más partidas sin mostrar tablero
    print(f"\n¿Quieres simular más partidas rápidas? (s/n)")
    try:
        if input().lower() == 's':
            num_games = int(input("¿Cuántas partidas? "))
            minimax_wins = 0
            random_wins = 0
            ties = 0
            
            for i in range(num_games):
                winner, _, _ = play_game(minimax_player, random_player, sizeBoard, show_board=False)
                if winner == "MAX":
                    minimax_wins += 1
                elif winner == "MIN":
                    random_wins += 1
                else:
                    ties += 1
                print(f"Partida {i+1}: {winner}")
            print("="*50)
            print(f"\nRESULTADOS DE {num_games} PARTIDAS:")
            print(f"   Minimax ganó: {minimax_wins} ({minimax_wins/num_games*100:.1f}%)")
            print(f"   Aleatorio ganó: {random_wins} ({random_wins/num_games*100:.1f}%)")
            print(f"   Empates: {ties} ({ties/num_games*100:.1f}%)")
    except:
        print("Finalizando programa...")