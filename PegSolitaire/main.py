import numpy as np
import heapq
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

Coord = Tuple[int, int]
Move = Tuple[Coord, Coord]   

# -----------------------------
# Tablero inicial
# -----------------------------
def initial_board() -> np.ndarray:
    # -1 representa una celda inválida, 1 una ficha, 0 un espacio vacío
    b = np.array([
        [-1, -1,  1,  1,  1, -1, -1],
        [-1, -1,  1,  1,  1, -1, -1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  0,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1],
        [-1, -1,  1,  1,  1, -1, -1],
        [-1, -1,  1,  1,  1, -1, -1],
    ])
    return b


# Cuenta el número de fichas en el tablero (valor de g)
def count_pegs(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 1))


# Verifica si se ha ganado: solo queda una ficha y está en el centro
def has_won(board: np.ndarray) -> bool:
    return count_pegs(board) == 1 and int(board[3, 3]) == 1


# Genera todos los movimientos legales posibles para el tablero dado
def get_possible_moves(board: np.ndarray) -> List[Move]:
    moves: List[Move] = []
    peg_rows, peg_cols = np.where(board == 1)
    for row, col in zip(peg_rows, peg_cols):
        # Izquierda a derecha (L->R): (1,1,0)
        if col <= 4 and board[row, col+1] == 1 and board[row, col+2] == 0:
            moves.append(((row, col), (row, col+2)))
        # Derecha a izquierda (R->L): (1,1,0)
        if col >= 2 and board[row, col-1] == 1 and board[row, col-2] == 0:
            moves.append(((row, col), (row, col-2)))
        # Arriba a abajo (U->D): (1,1,0)
        if row <= 4 and board[row+1, col] == 1 and board[row+2, col] == 0:
            moves.append(((row, col), (row+2, col)))
        # Abajo a arriba (D->U): (1,1,0)
        if row >= 2 and board[row-1, col] == 1 and board[row-2, col] == 0:
            moves.append(((row, col), (row-2, col)))
    return moves


# Aplica un movimiento al tablero y devuelve el nuevo tablero resultante
def apply_move(board: np.ndarray, move: Move) -> np.ndarray:
    (row1, col1), (row2, col2) = move
    rm = ((row1 + row2) // 2, (col1 + col2) // 2)  # Ficha eliminada
    newBoard = board.copy()
    newBoard[row1, col1] = 0
    newBoard[rm[0], rm[1]] = 0
    newBoard[row2, col2] = 1
    return newBoard


# Convierte el tablero a bytes para usarlo como clave en la lista cerrada
def board_key(board: np.ndarray) -> bytes:
    return board.tobytes()


# Imprime el tablero de forma entendible para humanos
def print_board(board: np.ndarray) -> None:
    trans = np.where(board == -1, ' ', np.where(board == 0, 'o', 'X'))
    print(" ---------------")
    for r in range(board.shape[0]):
        print("| " + "".join(trans[r].tolist()) + " |")
    print(" ---------------")



# Heurística para A*: número de fichas restantes menos 1
def heuristic(board: np.ndarray) -> int:
    return count_pegs(board) - 1


# Clase para nodos en A*
@dataclass(order=True)
class State:
    f: int  # Costo total estimado (g + h)
    h: int  # Heurística
    g: int  # Costo real desde el inicio
    board: np.ndarray  # Estado del tablero
    parent_idx: Optional[int]  # Índice del padre en la lista de estados
    move_from_parent: Optional[Move]  # Movimiento que llevó a este estado



# Reconstruye el camino de la solución desde el estado meta hasta el inicial
def reconstruct_path(states: List[State], goal_idx: int):
    moves: List[Move] = []
    boards: List[np.ndarray] = []
    costs: List[Tuple[int,int,int]] = [] 
    idx = goal_idx
    while idx is not None:
        n = states[idx]
        boards.append(n.board)
        costs.append((n.f, n.g, n.h))
        if n.move_from_parent is not None:
            moves.append(n.move_from_parent)
        idx = n.parent_idx
    moves.reverse()
    boards.reverse()
    costs.reverse()
    return moves, boards, costs



# Algoritmo A* para resolver el Peg Solitaire
def astar(start: np.ndarray, verbose: bool = True):

    states: List[State] = []  # Lista de todos los estados generados
    tie_counter = 0  # Desempate para el heap

    h0 = heuristic(start)
    start_state = State(f=h0, h=h0, g=0, board=start, parent_idx=None, move_from_parent=None)
    states.append(start_state)

    open_heap: List[Tuple[int, int, int, int, int]] = []  # (f,h,g,tie,idx)
    heapq.heappush(open_heap, (start_state.f, start_state.h, start_state.g, tie_counter, 0))

    closed_best_g: dict[bytes, int] = { board_key(start): 0 }

    expansions = 0  # Número de expansiones
    max_open = 1    # Máximo tamaño de la lista abierta

    start_time = time.time()
    found = False
    goal_result = None
    while open_heap:
        if verbose:
            max_open = max(max_open, len(open_heap))

        _, _, _, _, idx = heapq.heappop(open_heap)
        current = states[idx]
        expansions += 1

        # Verifica si se ha alcanzado el objetivo
        if has_won(current.board):
            found = True
            goal_result = reconstruct_path(states, idx)
            break

        # Genera todos los movimientos posibles desde el estado actual
        for mv in get_possible_moves(current.board):
            nxt = apply_move(current.board, mv)
            g2 = current.g + 1
            k = board_key(nxt)

            prev = closed_best_g.get(k)
            if prev is not None and prev <= g2:
                continue

            h2 = heuristic(nxt)
            nstate = State(f=g2 + h2, h=h2, g=g2, board=nxt, parent_idx=idx, move_from_parent=mv)
            states.append(nstate)
            new_idx = len(states) - 1
            tie_counter += 1
            closed_best_g[k] = g2
            heapq.heappush(open_heap, (nstate.f, nstate.h, nstate.g, tie_counter, new_idx))

    end_time = time.time()
    elapsed = end_time - start_time
    # Resultados de la búsqueda
    print("\n==============================")
    if found:
        print("Estado objetivo: SÍ")
    else:
        print("Estado objetivo: NO")
    print(f"Duración de la evaluación: {elapsed:.4f} segundos")
    print(f"Tamaño de la lista abierta: {len(open_heap)}")
    print(f"Tamaño de la lista cerrada: {len(closed_best_g)}")
    print(f"Número de estados evaluados: {expansions}")
    print("==============================\n")
    if found:
        return goal_result
    else:
        return None



# Imprime el camino de la solución paso a paso
def print_roadmap(moves: List[Move], boards: List[np.ndarray], costs: List[Tuple[int,int,int]]) -> None:
    print("\n=== ROADMAP TO SOLUTION ===\n")
    print("Estado inicial:")
    print_board(boards[0])
    print(f"Costos: f={costs[0][0]}, g={costs[0][1]}, h={costs[0][2]}")
    for i, (mv, b, (f,g,h)) in enumerate(zip(moves, boards[1:], costs[1:]), start=1):
        (r1, c1), (r2, c2) = mv
        print(f"Paso {i}: mover {(int(r1), int(c1))} -> {(int(r2), int(c2))}")
        print_board(b)
        print(f"Costos: f={f}, g={g}, h={h}")
    print(f"Total de pasos: {len(moves)}")
    print(f"¿Ganó? {has_won(boards[-1])}")


# -----------------------------
# Ejecución de demostración
# -----------------------------
if __name__ == "__main__":
    board = initial_board()
    result = astar(board, verbose=True)
    if result is None:
        print("No es resoluble desde este inicio.")
    else:
        moves, boards, costs = result    
        print(f"\nLongitud de la solución = {len(moves)}")
        print_roadmap(moves, boards, costs)

