"""A simple console based Tic-Tac-Toe game"""

import random

import numpy as np

BOARD_SIZE = 3


def main() -> None:
    """Executes the game loop."""
    your_turn: bool = random.choice([True, False])
    game_is_running: bool = True
    game_board = np.zeros((3, 3), dtype=int)
    row: int = 0
    column: int = 0
    someone_has_won: bool = False
    winner: str = ""
    perspective_player: int = 2
    while game_is_running:
        draw_game_board(game_board)
        current_player: int = 1 if your_turn else 2
        # encoded_board = encode_board(game_board, current_player)
        if your_turn:
            move_possible: bool = False
            while not move_possible:
                move_possible, row, column = is_move_possible(
                    input(
                        "Please enter valid coordinates. Syntax: "
                        "row column (starting with 0)"
                    ),
                    game_board,
                )
            handle_inputs(row, column, game_board, 1)
        else:
            calculate_best_move(game_board, current_player, perspective_player)
        your_turn = not your_turn
        someone_has_won, winner = declare_winner(game_board)
        if someone_has_won:
            draw_game_board(game_board)
            game_is_running = False
            if winner == "1":
                winner = "X"
            elif winner == "2":
                winner = "O"
            else:
                winner = "Draw"
    print(f"The winner is: {winner}")


def draw_game_board(game_board) -> None:
    """Draws the game board."""

    row_buffer: list[str] = []
    for i, value in enumerate(game_board.flat):
        if value == 0:
            row_buffer.append(" ")
        elif value == 1:
            row_buffer.append("X")
        else:
            row_buffer.append("O")

        if (i + 1) % 3 == 0:
            print("|".join(row_buffer))
            row_buffer = []


def is_move_possible(coordinates: str, game_board) -> tuple[bool, int, int]:
    """Checks if the move is possible"""
    try:
        row, column = map(int, coordinates.split())
    except ValueError:
        return False, 0, 0

    if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE):
        return False, 0, 0

    if game_board[row, column] != 0:
        return False, 0, 0

    return True, row, column


def handle_inputs(row: int, column: int, game_board, symbol: int) -> None:
    """Executes the move"""
    game_board[row, column] = symbol


def calculate_best_move(game_board, current_player, perspective_player) -> None:
    """Does the best possible move"""
    _, index = minimax(game_board, current_player, perspective_player)
    if index is None:
        return
    row = index // 3
    column = index % 3
    game_board[row, column] = current_player


def declare_winner(game_board) -> tuple[bool, str]:
    """Returns if there is a winner and if yes, who"""
    for i in range(3):
        if (
            game_board[i, 0] == game_board[i, 1]
            and game_board[i, 1] == game_board[i, 2]
        ) and (game_board[i, 0] != 0):
            return True, str(game_board[i, 0])
        if (
            game_board[0, i] == game_board[1, i]
            and game_board[1, i] == game_board[2, i]
        ) and (game_board[0, i] != 0):
            return True, str(game_board[0, i])
    if (
        (game_board[1, 1] == game_board[2, 2] and game_board[2, 2] == game_board[0, 0])
        or (
            game_board[0, 2] == game_board[1, 1]
            and game_board[1, 1] == game_board[2, 0]
        )
    ) and (game_board[1, 1] != 0):
        return True, str(game_board[1, 1])
    if np.all(game_board != 0):
        return True, "Draw"
    return False, ""


def encode_board(game_board, current_player: int) -> list[int]:
    """Encodes the current Game Board for the neural network"""
    encoded_game_board: list[int] = []

    for row in range(3):
        for column in range(3):
            cell = game_board[row, column]

            if cell == 0:
                encoded_game_board.append(0)
            elif cell == current_player:
                encoded_game_board.append(1)
            else:
                encoded_game_board.append(-1)

    return encoded_game_board


def get_legal_moves(game_board) -> list[int]:
    """Checks for every legal move"""
    legal_indexes: list[int] = []
    index: int = 0
    for row in range(3):
        for column in range(3):
            if game_board[row, column] == 0:
                index = row * 3 + column
                legal_indexes.append(index)
    return legal_indexes


def get_terminal_score(game_board, perspective_player: int) -> int | None:
    """Evaluates the position form the current players perspective."""
    someone_has_won, winner = declare_winner(game_board)

    if not someone_has_won:
        return None
    if winner == "Draw":
        return 0
    if winner == str(perspective_player):
        return 1
    return -1


def minimax(
    game_board, player_to_move: int, perspective_player: int
) -> tuple[int, int | None]:
    """Maximizes the score for the compute while minimzing player score."""
    score = get_terminal_score(game_board, perspective_player)

    if score is not None:
        return score, None

    legal_indexes: list[int] = get_legal_moves(game_board)
    maximize: bool = player_to_move == perspective_player
    best_score: int | None = None
    best_move: int | None = None

    for move_index in legal_indexes:
        new_board = game_board.copy()
        row = move_index // 3
        column = move_index % 3
        new_board[row, column] = player_to_move
        next_player = 2 if player_to_move == 1 else 1
        current_score, _ = minimax(new_board, next_player, perspective_player)

        if (
            best_score is None
            or (maximize and current_score > best_score)
            or (not maximize and current_score < best_score)
        ):
            best_score = current_score
            best_move = move_index

    if best_score is None:
        return 0, None
    return best_score, best_move


if __name__ == "__main__":
    main()
