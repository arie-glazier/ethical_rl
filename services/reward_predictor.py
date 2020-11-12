import pickle, sys, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# TODO: i dont like this organization at all, need to redo
class RewardPredictor:
  def __init__(self, **kwargs):
    self.min_mu = kwargs["min_mu"]
    self.max_mu = kwargs["max_mu"]

    # we need to one hot encode states and actions so input is the same
    # here as it will be for agent training.
    # TODO: add state & env info to training output so we can just grab whatever
    # we happened to use.  now its [agent position, direction] + [action]
    self.max_x = 3 # width - buffer_size
    self.max_y = 3 # height - buffer_size
    self.n_directions = 4 # n_directions
    self.n_actions = 3 # can be more if add actions
    self.input_shape = self.n_directions + self.max_x + self.max_y + self.n_actions 

    self.training_data = [] # TODO: this mutable list is not good
    self.training_labels = []

  def mu(self, label):
    # TODO: standardize labels between human and synthetic
    if label == "1" or "left": return (self.max_mu, self.min_mu)
    elif label == "2" or "right": return (self.min_mu, self.max_mu)
    elif label == "3" or "same": return ((self.max_mu + self.min_mu) / 2, (self.max_mu + self.min_mu) / 2)
    else: return None

  def process_trajectory(self, trajectory, mu_idx, label):
    for state, action in trajectory:
      directions = np.zeros(self.n_directions)
      directions[state["direction"]] = 1

      agent_position = state["agent_position"]
      x_position = np.zeros(self.max_x)
      x_position[agent_position[0] - 1] = 1

      y_position = np.zeros(self.max_y)
      y_position[agent_position[1]-1] = 1

      actions = np.zeros(self.n_actions)
      actions[action] = 1

      input_array = np.concatenate((
        directions,
        x_position,
        y_position,
        actions
      ))

      self.training_data.append(input_array)
      self.training_labels.append(self.mu(label)[mu_idx]) # TODO: ugly

  def train_model(self):
    x_train_all, x_test, y_train_all, y_test = train_test_split(np.stack(self.training_data), np.array(self.training_labels), train_size=0.9)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, train_size=0.9)
    # TODO: update models so we can just use them here
    input_layer = keras.layers.Dense(30, activation="elu", input_shape=(self.input_shape,))
    hidden_layer = keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))
    output_layer = keras.layers.Dense(1)
    model = keras.models.Sequential([
      input_layer,
      keras.layers.Dropout(rate=0.2),
      hidden_layer,
      keras.layers.Dropout(rate=0.2),
      output_layer
    ])

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid,y_valid))
    mse_test = model.evaluate(x_test, y_test)
    print(f"mse: {mse_test}")

    # TODO: config, set default folders on deploy
    model.save("./saved_models/reward_model.h5")

    return model

  def predict(self, model, direction, x, y, action):
    directions = np.zeros(self.n_directions)
    directions[direction] = 1

    x_position = np.zeros(self.max_x)
    x_position[x] = 1

    y_position = np.zeros(self.max_y)
    y_position[y] = 1

    actions = np.zeros(self.n_actions)
    actions[action] = 1

    input_array = np.concatenate((
        directions,
        x_position,
        y_position,
        actions
      ))

    print(input_array)

    prediction = model.predict(np.vstack([input_array]))
    print(f"starting state ({x}, {y}) => action: {action}, prediction: {prediction}")
