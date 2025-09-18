def print_board(game):
    """Dibuja el tablero en consola."""
    size = game.size
    print("="*40)
    for r in range(size+1):
        line = "•"
        for c in range(size):
            line += "---•" if game.horizontal_lines[r][c] else "   •"
        print(line)
        if r < size:
            line = ""
            for c in range(size+1):
                line += "|" if game.vertical_lines[r][c] else " "
                if c < size:
                    line += " X " if game.completed_boxes[r][c] else "   "
            print(line)
    print("="*40)
    print(f"Scores → MAX: {game.scores[0]} | MIN: {game.scores[1]}")
    print(f"Turno actual: {'MAX' if game.current_player==0 else 'MIN'}")
    print("="*40)
