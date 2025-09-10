from game import DotsAndBoxes
from players import MinimaxPlayer, RandomPlayer
from utils import print_board

def play_game(p1, p2, size, show=True):
    """Ejecuta una partida completa entre dos jugadores."""
    game = DotsAndBoxes(size)
    players = [p1, p2]
    print("=== INICIANDO PARTIDA ===")
    while not game.is_game_over():
        player = players[game.current_player]
        move = player.get_best_move(game)
        if not move: break
        game.make_move(*move)
        if show:
            print_board(game)
            input("Presiona Enter para continuar...")

    print("=== RESULTADO FINAL ===")
    print_board(game)

if __name__ == "__main__":
    minimax = MinimaxPlayer(max_depth=5)
    random_player = RandomPlayer()
    play_game(minimax, random_player, size=2, show=True)
