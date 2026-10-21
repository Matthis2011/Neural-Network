"""Creates the tic_tac_toe training data for the neural nbetwork"""

import json
from pathlib import Path

import numpy as np

import tic_tac_toe

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "tic_tac_toe_training_data.json"
)


def main():
    """main function that creates the training data"""
    board = np.zeros((3, 3), dtype=int)
    save_training_data(board)


def encode_board(board, player):
    """
    converts board into 9 input values.
    0 = empty
    1 = player
    -1 = enemy
    """
    return tic_tac_toe.encode_board(board, player)


def build_target_vector(best_move_indices: list[int]) -> list[float]:
    """
    Distributes the target probability equally across all equally best moves.
    """
    if not best_move_indices:
        raise ValueError("At least one best move is required.")

    targets = [0.0] * 9
    target_probability = 1 / len(best_move_indices)
    for move_index in best_move_indices:
        targets[move_index] = target_probability
    return targets


def get_best_moves(board, player_to_move: int, perspective_player: int) -> list[int]:
    """Returns all legal moves with the best minimax score for the current player."""
    board = np.asarray(board, dtype=int)
    score = tic_tac_toe.get_terminal_score(board, perspective_player)
    if score is not None:
        return []

    legal_indexes: list[int] = get_legal_moves(board)
    maximize: bool = player_to_move == perspective_player
    best_score: int | None = None
    best_moves: list[int] = []

    for move_index in legal_indexes:
        new_board = board.copy()
        row = move_index // 3
        column = move_index % 3
        new_board[row, column] = player_to_move
        next_player = 2 if player_to_move == 1 else 1
        current_score, _ = tic_tac_toe.minimax(
            new_board, next_player, perspective_player
        )

        if best_score is None:
            best_score = current_score
            best_moves = [move_index]
            continue

        if maximize and current_score > best_score:
            best_score = current_score
            best_moves = [move_index]
        elif maximize and current_score == best_score:
            best_moves.append(move_index)
        elif (not maximize) and current_score < best_score:
            best_score = current_score
            best_moves = [move_index]
        elif (not maximize) and current_score == best_score:
            best_moves.append(move_index)

    return best_moves


def get_legal_moves(board) -> list[int]:
    """
    returns indexes of all free spots on the game board
    """
    indexes = tic_tac_toe.get_legal_moves(board)
    return indexes


def generate_examples(board, player_to_move: int, seen_states=None):
    """
    iterates recursively over all possible moves and returns all possible game boards
    """
    board = np.asarray(board, dtype=int)
    if seen_states is None:
        seen_states = set()
    examples = []
    state_key = (
        tuple(int(value) for value in board.flatten()),
        player_to_move,
    )
    if state_key in seen_states:
        return []
    seen_states.add(state_key)
    someone_has_won, _ = tic_tac_toe.declare_winner(board)
    if someone_has_won:
        return examples

    best_move_indices = get_best_moves(board, player_to_move, player_to_move)
    if not best_move_indices:
        return examples
    examples.append(
        {
            "inputs": encode_board(board, player_to_move),
            "targets": build_target_vector(best_move_indices),
        }
    )

    possible_moves: list[int] = get_legal_moves(board)
    for move_index in possible_moves:
        next_board = board.copy()
        row = move_index // 3
        column = move_index % 3
        next_board[row, column] = player_to_move
        next_player = 2 if player_to_move == 1 else 1
        examples.extend(generate_examples(next_board, next_player, seen_states))
    return examples


def save_training_data(
    board,
    path: Path = DATA_PATH,
) -> None:
    """Generate and save Tic-Tac-Toe training examples."""
    data = generate_examples(board, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"training examples saved in {path}")


if __name__ == "__main__":
    main()
