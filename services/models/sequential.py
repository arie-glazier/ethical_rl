import sys
from tensorflow import keras

class SequentialModel:
  def __init__(self, **kwargs):
    self.env = kwargs["environment"]

    # this is the structure of obs. space https://github.com/openai/gym/issues/593
    # EX: cartpole = [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    self.input_shape = self.env.observation_space.shape 

    self.n_outputs = self.env.action_space.n

  def simple_model(self):
    return keras.models.Sequential([
      keras.layers.Dense(32, activation="elu", input_shape=self.input_shape),
      keras.layers.Dense(32, activation="elu"),
      keras.layers.Dense(self.n_outputs)
    ])