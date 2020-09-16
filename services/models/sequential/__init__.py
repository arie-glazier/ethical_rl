from services.constants import *

class SequentialModelBASE:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]

    # this is the structure of obs. space https://github.com/openai/gym/issues/593
    # EX: cartpole = [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    self.input_shape = self.env.observation_space.shape

    self.n_outputs = self.env.action_space.n