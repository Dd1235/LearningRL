import pickle
import random

from engine import Engine

# Value function for each state (MDP)
mdp = dict()
epsilon = 0.1
alpha = 0.1  # learning rate


def train(num_episodes=100000):
    mdp["000000000"] = 0

    for _ in range(num_episodes):  # generate episodes
        engine = Engine()

        while not engine.is_game_over():
            # while game is not over continue playing
            board = engine.get_board_string()

            if board not in mdp:
                mdp[board] = 0

            # generate all possible moves possible
            children = engine.get_children_strings()
            child_values = {}

            for child in children:  # for each possible move
                if child not in mdp:  # if I have not done it
                    winner = engine.check_child_win(child)
                    if winner == "X":
                        mdp[child] = 1  # reward for win
                    elif winner == "O":
                        mdp[child] = -1  # penalty for loss
                    else:
                        mdp[child] = 0  # neutral

                child_values[child] = mdp[child]

            # ε-greedy policy , is the random fraction less than epsilon
            if random.random() < epsilon:
                move = random.choice(
                    engine.get_valid_moves()
                )  # exploration, make some random move
                engine.make_move(move)
                next_state = engine.get_board_string()
                mdp[board] += alpha * (mdp[next_state] - mdp[board])
            else:  # exploitation

                best_child = max(child_values, key=child_values.get)
                move = engine.get_move_from_child(best_child)
                engine.make_move(move)
                mdp[board] += alpha * (mdp[best_child] - mdp[board])

            # --- Opponent makes a random move (off-policy update idea) ---
            if not engine.is_game_over():
                engine.make_move(random.choice(engine.get_valid_moves()))

    with open("mdp_values.pkl", "wb") as f:
        pickle.dump(mdp, f)


if __name__ == "__main__":
    train()
