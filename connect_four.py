"""A simple console based connect four game."""

import copy
import random
from pathlib import Path

import numpy as np

import neural_network

PATH = Path(__file__).parent / "models" / "connect_four_model.json"


def main() -> None:
    """Executes the game loop."""
    network = neural_network.NeuralNetwork.load(PATH)
    your_turn: bool = random.choice([True, False])
    game_is_running: bool = True
    game_board = np.zeros((6, 7), dtype=int)
    row: int | None = None
    column: int | None = None
    someone_has_won: bool = False
    winner: str = ""
    while game_is_running:
        draw_game_board(game_board)
        current_player: int = 1 if your_turn else 2
        if your_turn:
            move_possible: bool = False
            while not move_possible:
                column_choice = int(
                    input("Please enter a valid column (starting with 0)")
                )
                move_possible, row, column = is_move_possible(column_choice, game_board)
                if move_possible and row is not None and column is not None:
                    handle_inputs(row, column, game_board, current_player)
        else:
            ai_move(network, game_board, current_player)
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
    """Draws the game game_board."""
    print("")
    row_buffer: list[str] = []
    for i, value in enumerate(game_board.flat):
        if value == 0:
            row_buffer.append(" ")
        elif value == 1:
            row_buffer.append("X")
        else:
            row_buffer.append("O")

        if (i + 1) % 7 == 0:
            print("|".join(row_buffer))
            row_buffer = []


def is_move_possible(column: int, game_board) -> tuple[bool, int | None, int | None]:
    """Checks if the move is possible"""
    six: int = 6
    if column > six or column < 0:
        return False, None, None
    if game_board[0, column] != 0:
        return False, None, None
    for current_row in range(5, -1, -1):
        if game_board[current_row, column] == 0:
            return True, current_row, column
    return False, None, None


def handle_inputs(row: int, column: int, game_board, symbol: int) -> None:
    """Executes the move"""

    game_board[row, column] = symbol


def ai_move(network, game_board, current_player) -> None:
    """Execute AIs choice"""
    column = ai_choice(network, encode_board(game_board, current_player), game_board)
    if column is None:
        return
    move_possible, row, column = is_move_possible(column, game_board)
    if move_possible and row is not None and column is not None:
        handle_inputs(row, column, game_board, current_player)


def declare_winner(game_board) -> tuple[bool, str]:
    """Returns if there is a winner and if yes, who"""
    columns = 7
    rows = 6
    board_full: bool = True

    for i in range(6):
        for j in range(4):
            symbol = game_board[i, j]
            if (
                symbol != 0
                and symbol
                == game_board[i, j + 1]
                == game_board[i, j + 2]
                == game_board[i, j + 3]
            ):
                return True, str(symbol)

    for i in range(3):
        for j in range(7):
            symbol = game_board[i, j]
            if (
                symbol != 0
                and symbol
                == game_board[i + 1, j]
                == game_board[i + 2, j]
                == game_board[i + 3, j]
            ):
                return True, str(symbol)

    for i in range(rows - 3):
        for j in range(columns - 3):
            symbol = game_board[i, j]
            if (
                symbol != 0
                and symbol
                == game_board[i + 1, j + 1]
                == game_board[i + 2, j + 2]
                == game_board[i + 3, j + 3]
            ):
                return True, str(symbol)

    for i in range(rows - 3):
        for j in range(3, columns):
            symbol = game_board[i, j]
            if (
                symbol != 0
                and symbol
                == game_board[i + 1, j - 1]
                == game_board[i + 2, j - 2]
                == game_board[i + 3, j - 3]
            ):
                return True, str(symbol)

    for i in range(rows):
        for j in range(columns):
            if game_board[i, j] == 0:
                board_full = False
    if board_full:
        return True, "Draw"
    return False, ""


def encode_board(game_board, current_player: int) -> list[int]:
    """Encodes the current game board for the neural network"""
    encoded_game_board: list[int] = []

    for row in range(6):
        for column in range(7):
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
    legal_columns: list[int] = []
    column: int = 0
    for column in range(7):
        if game_board[0, column] == 0:
            legal_columns.append(column)
    return legal_columns


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


def ai_choice(network, encoded_board, game_board):
    """Pass the current game board to the MLP."""
    predictions = neural_network.softmax(network.forward(encoded_board))
    legal_moves = get_legal_moves(game_board)
    if not legal_moves:
        return None
    move_index = max(
        legal_moves, key=lambda first_player_index: predictions[first_player_index]
    )
    return move_index


def play_game(network_1, network_2, first_player=1):
    """Simulate a game between two neural networks."""
    game_board = np.zeros((6, 7), dtype=int)
    networks = {1: network_1, 2: network_2}
    current_player = first_player

    while True:
        draw_game_board(game_board)
        network = networks[current_player]
        encoded_board = encode_board(game_board, current_player)
        column = ai_choice(network, encoded_board, game_board)

        if column is None:
            return 0

        move_possible, row, column = is_move_possible(column, game_board)

        if not move_possible or row is None or column is None:
            raise RuntimeError("AI selected an illegal move.")

        handle_inputs(row, column, game_board, current_player)

        someone_has_won, winner = declare_winner(game_board)

        if someone_has_won:
            draw_game_board(game_board)
            if winner == "Draw":
                return 0
            return int(winner)

        current_player = 3 - current_player


def evaluate_fitness(player_1_index, player_2_index, fitnesses, winner):
    """Evaluate fitnesses of neural networks based on their games outcome."""
    if winner == 0:
        fitnesses[player_1_index] += 0.5
        fitnesses[player_2_index] += 0.5
    else:
        fitnesses[player_1_index if winner == 1 else player_2_index] += 1


def ai_game_loop(population, population_size, pool, first_player, fitnesses):
    """Pair every neural network against another network and run the game."""
    for i in range(0, population_size, 2):
        first_player_index = pool[i]
        second_player_index = pool[i + 1]
        result = play_game(
            population[first_player_index],
            population[second_player_index],
            first_player,
        )
        print(str(result))
        evaluate_fitness(first_player_index, second_player_index, fitnesses, result)
        print(
            str(fitnesses[first_player_index])
            + "\n"
            + str(fitnesses[second_player_index])
        )


def select_parents(population=None):
    """Let the neural_networks play against each other and choose the best of them."""
    population_size = 100
    fitnesses = [0 for _ in range(population_size)]
    if population is None:
        population = [
            neural_network.NeuralNetwork(
                input_size=42, hidden_sizes=[64, 64], output_size=7
            )
            for _ in range(population_size)
        ]
    pool = list(range(population_size))
    for _ in range(10):
        random.shuffle(pool)
        ai_game_loop(population, population_size, pool, 1, fitnesses)
        ai_game_loop(population, population_size, pool, 2, fitnesses)
    top_ten = sorted(pool, key=lambda i: fitnesses[i], reverse=True)[:10]
    new_population = [copy.deepcopy(population[top_ten[i]]) for i in range(2)]
    return top_ten, new_population, population


def create_child(top_ten, population):
    """Create a single child-neural network with the parameters of its parents."""
    index, index_2 = random.sample(range(10), 2)
    parent_1 = population[top_ten[index]]
    parent_2 = population[top_ten[index_2]]
    child = neural_network.NeuralNetwork(
        parent_1.input_size, parent_1.hidden_sizes, parent_1.output_size, None
    )
    for i, layer in enumerate(parent_1.hidden_layers):
        for j, neuron in enumerate(layer.neurons):
            child.hidden_layers[i].neurons[j].bias = (
                parent_1.hidden_layers[i].neurons[j].bias
                if random.choice((1, 2)) == 1
                else parent_2.hidden_layers[i].neurons[j].bias
            )
            for k, _ in enumerate(neuron.weights):
                child.hidden_layers[i].neurons[j].weights[k] = (
                    parent_1.hidden_layers[i].neurons[j].weights[k]
                    if random.choice((1, 2)) == 1
                    else parent_2.hidden_layers[i].neurons[j].weights[k]
                )

    for l, neuron in enumerate(parent_1.output_layer.neurons):
        child.output_layer.neurons[l].bias = (
            parent_1.output_layer.neurons[l].bias
            if random.choice((1, 2)) == 1
            else parent_2.output_layer.neurons[l].bias
        )
        for m, _ in enumerate(neuron.weights):
            child.output_layer.neurons[l].weights[m] = (
                parent_1.output_layer.neurons[l].weights[m]
                if random.choice((1, 2)) == 1
                else parent_2.output_layer.neurons[l].weights[m]
            )
    return mutate_network(child)


def mutate_network(network):
    """Mutate every singel paramter of the network with a 5% chance."""
    for _, layer in enumerate(network.hidden_layers):
        for _, neuron in enumerate(layer.neurons):
            if random.randrange(20) == 1:
                neuron.bias += random.gauss(0, 0.1)
            for i, _ in enumerate(neuron.weights):
                if random.randrange(20) == 1:
                    neuron.weights[i] += random.gauss(0, 0.1)
    for _, neuron in enumerate(network.output_layer.neurons):
        if random.randrange(20) == 1:
            neuron.bias += random.gauss(0, 0.1)
        for j, _ in enumerate(neuron.weights):
            if random.randrange(20) == 1:
                neuron.weights[j] += random.gauss(0, 0.1)
    return network


def read_parameters():
    """Read the current paramteres form .json file."""
    return []


def save_parameters():
    """Saves the current parameters as a .json file."""
    return


def evolutionary_training(generations):
    """Train the ai using the principles of evolution."""
    population = read_parameters()
    for _ in range(generations):
        top_ten, new_population, population = select_parents(population)
        for _ in range(98):
            new_population.append(create_child(top_ten, population))
        population = new_population
    save_parameters()


if __name__ == "__main__":
    evolutionary_training(int(input("enter number of populations: ")))
    # main()
