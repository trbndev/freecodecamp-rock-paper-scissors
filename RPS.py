import random
import numpy as np
from tensorflow import keras

play_encoding = {"R": 0, "P": 1, "S": 2}
play_decoding = {value: key for key, value in play_encoding.items()}


def play_to_int(play):
    return play_encoding[play]


def int_to_play(play):
    return play_decoding[play]


def prepare_data(plays, length=15):
    X, y = [], []
    for i in range(len(plays) - length):
        X.append(plays[i : i + length])  # Append the next X plays
        y.append(plays[i + length])  # Append the play after the X plays
    return np.array(X), np.array(y)


model = keras.models.Sequential(
    [
        keras.layers.Embedding(input_dim=3, output_dim=8),
        keras.layers.LSTM(32),
        keras.layers.Dense(3, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)


def train_model(X, y, log_progress):
    global model

    _, accuracy = model.evaluate(X, y, verbose=0)

    if accuracy > 0.99:
        return

    model.fit(X, y, epochs=10, verbose=log_progress)


def player(prev_play, opponent_history=[]):
    global model, play_history

    # Convert the opponent's play to an integer
    if prev_play != "":
        opponent_history.append(play_to_int(prev_play))

    # Gather data until we have enough to make a prediction
    if len(opponent_history) < 16:
        return random.choice(["R", "P", "S"])

    X, y = prepare_data(opponent_history)

    if len(opponent_history) % 10 == 0:
        log_progress = 1 if len(opponent_history) % 100 == 0 else 0
        train_model(X, y, log_progress)

    predictions = model.predict(X[-1].reshape(1, 15), verbose=0)[0]
    next_play = np.argmax(predictions)

    return counter_play(next_play)


def counter_play(play):
    decoded_play = int_to_play(play)
    counters = {"R": "P", "P": "S", "S": "R"}
    return counters[decoded_play]
