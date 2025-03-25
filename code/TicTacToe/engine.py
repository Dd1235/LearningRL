import copy
import random


class Engine:
    def __init__(self):
        self.board = [["0"] * 3 for _ in range(3)]
        self.turn = "X"  # X starts by default

    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == "0"]

    def make_move(self, move):
        i, j = move
        self.board[i][j] = self.turn
        self.turn = "O" if self.turn == "X" else "X"

    def undo_move(self, move):
        i, j = move
        self.board[i][j] = "0"
        self.turn = "O" if self.turn == "X" else "X"

    def check_winner(self):

        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != "0":
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != "0":
                return self.board[0][i]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != "0":
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != "0":
            return self.board[0][2]
        return "0"

    def is_game_over(self):
        return self.check_winner() != "0" or len(self.get_valid_moves()) == 0

    def get_board_string(self):
        return "".join("".join(row) for row in self.board)

    def get_children_strings(self):

        children = []
        for move in self.get_valid_moves():
            self.make_move(move)
            children.append(self.get_board_string())
            self.undo_move(move)
        return children

    @staticmethod
    def check_child_win(child_string):

        for i in range(3):
            if (
                child_string[i * 3 : i * 3 + 3].count(child_string[i * 3]) == 3
                and child_string[i * 3] != "0"
            ):
                return child_string[i * 3]
            if child_string[i] == child_string[i + 3] == child_string[i + 6] != "0":
                return child_string[i]
        if child_string[0] == child_string[4] == child_string[8] != "0":
            return child_string[0]
        if child_string[2] == child_string[4] == child_string[6] != "0":
            return child_string[2]
        return "0"

    def get_move_from_child(self, child_string):

        current = self.get_board_string()
        for i in range(9):
            if current[i] != child_string[i]:
                return (i // 3, i % 3)
