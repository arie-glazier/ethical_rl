from ethical_rl.constants import *

class SequentialModelBASE:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]

    self.input_shape = self.env.observation_space.shape
    self.n_outputs = self.env.action_space.n

    self.fully_connected_model_size = kwargs.get(FULLY_CONNECTED_MODEL_SIZE) or [100,100]