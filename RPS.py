import numpy as np
from itertools import product

# Hyperparameters
NUM_PREVIOUS_PLAYS = 2
LEARNING_RATE = 0.65
DISCOUNT_FACTOR = 0.5
EXPLORATION_RATE = 0.01

# Constants
Q_table = np.random.uniform(0, 1, size=[3 ** (2 * NUM_PREVIOUS_PLAYS), 3])

play_encoding = {"R": 0, "P": 1, "S": 2}
play_decoding = {value: key for key, value in play_encoding.items()}

play_states = list(product("RPS", repeat=2 * NUM_PREVIOUS_PLAYS))
play_state_encoding = {state: i for i, state in enumerate(play_states)}


# Player function
def player(prev_play, opponent_history=[], player_history=[]):
    opponent_history.append(prev_play)

    # First play is called with prev_play = ""
    is_first_play = "" in opponent_history[-NUM_PREVIOUS_PLAYS - 1 :]

    if is_first_play:
        # Choose random play
        play_encoded = choose_play(Q_table, None, EXPLORATION_RATE)
        play = play_decoding[play_encoded]

        # Fill Q-Table with random values
        Q_table[:, :] = np.random.uniform(0, 1, size=[3 ** (2 * NUM_PREVIOUS_PLAYS), 3])

        player_history.append(play)

        return play

    # Encode the previous and current states
    previous_state = tuple(
        player_history[-NUM_PREVIOUS_PLAYS - 1 : -1]
        + opponent_history[-NUM_PREVIOUS_PLAYS - 1 : -1]
    )
    previous_state_encoded = play_state_encoding[previous_state]
    previous_play_encoded = play_encoding[player_history[-1]]
    current_state = tuple(
        player_history[-NUM_PREVIOUS_PLAYS:] + opponent_history[-NUM_PREVIOUS_PLAYS:]
    )
    current_state_encoded = play_state_encoding[current_state]

    # Calculate reward based on the previous play
    reward = calculate_reward(player_history[-1], prev_play)

    # Update Q-table based on the reward and the current state
    Q_table[previous_state_encoded, previous_play_encoded] += LEARNING_RATE * (
        reward
        + DISCOUNT_FACTOR * np.max(Q_table[current_state_encoded])
        - Q_table[previous_state_encoded, previous_play_encoded]
    )

    # Choose the next play based on the Q-table and exploration rate
    play_encoded = choose_play(Q_table, current_state_encoded, EXPLORATION_RATE)
    play = play_decoding[play_encoded]

    # Keep track of the player's history
    player_history.append(play)

    return play


# Utility functions
def choose_play(Q_table, state_index, exploration_rate):
    # Generate a random number to decide between exploration and exploitation
    random_number = np.random.uniform(0, 1)

    # If random number is less than exploration rate, choose random play
    if random_number < exploration_rate or state_index is None:
        return np.random.choice(range(3))

    # Otherwise, choose the best play based on the Q-table
    return np.argmax(Q_table[state_index, :])


def calculate_reward(player, opponent):
    if player == opponent:
        return 0.0  # Tie situation, no reward
    elif (play_encoding[player] - play_encoding[opponent]) % 3 == 1:
        return 1.0  # Win situation, positive reward

    return -1.0  # Loss situation, negative reward
