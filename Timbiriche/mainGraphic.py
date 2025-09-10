import matplotlib.pyplot as plt
from game import DotsAndBoxes
from players import MinimaxPlayer, RandomPlayer
from utils import print_board

def play_game(p1, p2, size, show=True, collect_stats=False):
    game = DotsAndBoxes(size)
    players = [p1, p2]
    times = [[], []]  # tiempos por jugador
    nodes = [[], []]  # nodos por jugador
    while not game.is_game_over():
        player_idx = game.current_player
        player = players[player_idx]
        import time
        start = time.time()
        move = player.get_best_move(game)
        elapsed = time.time() - start
        if not move: break
        game.make_move(*move)
        if collect_stats and hasattr(player, "nodes_explored"):
            times[player_idx].append(elapsed)
            nodes[player_idx].append(player.nodes_explored)
        if show:
            print_board(game)
            input("Presiona Enter para continuar...")
    if collect_stats:
        return times, nodes, game.scores
    return None

if __name__ == "__main__":
    minimax = MinimaxPlayer(max_depth=3)
    random_player = RandomPlayer()
    # random_player = RandomPlayer()
    num_games = 10

    all_times = [[], []]
    all_nodes = [[], []]
    minimax_wins = 0
    minimax_scores = []
    random_scores = []
    for _ in range(num_games):
        times, nodes, scores = play_game(minimax, random_player, size=3, show=False, collect_stats=True)
        for i in [0, 1]:
            all_times[i].extend(times[i])
            all_nodes[i].extend(nodes[i])
        minimax_scores.append(scores[0])
        random_scores.append(scores[1])
        if scores[0] > scores[1]:
            minimax_wins += 1
    # Graficar
    plt.figure(figsize=(16,4))
    plt.subplot(1,4,1)
    plt.plot(all_times[0], marker='o', color='blue', label='Minimax')
    plt.plot(all_times[1], marker='o', color='red', label='Random')
    plt.title("Tiempo por jugada")
    plt.xlabel("Jugada")
    plt.ylabel("Segundos")
    plt.legend()
    plt.subplot(1,4,2)
    plt.plot(all_nodes[0], marker='o', color='blue', label='Minimax')
    plt.plot(all_nodes[1], marker='o', color='red', label='Random')
    plt.title("Nodos explorados por jugada")
    plt.xlabel("Jugada")
    plt.ylabel("Nodos")
    plt.legend()
    plt.subplot(1,4,3)
    plt.bar(["Minimax", "Random"], [minimax_wins, num_games-minimax_wins], color=['blue','red'])
    plt.title("Tasa de éxito (partidas ganadas)")
    plt.ylabel("Partidas ganadas")
    plt.subplot(1,4,4)
    plt.plot(minimax_scores, marker='o', color='blue', label='Minimax')
    plt.plot(random_scores, marker='o', color='red', label='Random')
    plt.title("Puntaje final por partida")
    plt.xlabel("Partida")
    plt.ylabel("Puntaje")
    plt.legend()
    plt.tight_layout()
    plt.show()