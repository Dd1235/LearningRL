import pickle

from engine import Engine

with open("mdp_values.pkl", "rb") as f:
    mdp = pickle.load(f)


def print_board(board):
    for row in board:
        print(" ".join(row))
    print()


def play():
    engine = Engine()
    print("You're O. AI is X. You play second.")
    print_board(engine.board)

    while not engine.is_game_over():
        # AI move (greedy)
        children = engine.get_children_strings()
        best_child = max(children, key=lambda c: mdp.get(c, 0))
        move = engine.get_move_from_child(best_child)
        engine.make_move(move)

        print("AI Move:")
        print_board(engine.board)

        if engine.is_game_over():
            break

        valid = False
        while not valid:
            try:
                i, j = map(int, input("Enter your move (i j): ").split())
                if (i, j) not in engine.get_valid_moves():
                    print("Invalid move! That cell is already taken or out of range.")
                else:
                    engine.make_move((i, j))
                    valid = True
            except Exception:
                print("Invalid input! Please enter two integers like: 0 2")

        print_board(engine.board)

    winner = engine.check_winner()
    if winner == "X":
        print("AI Wins!")
    elif winner == "O":
        print("You Win!")
    else:
        print("Draw!")


if __name__ == "__main__":
    play()
