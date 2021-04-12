from ethical_rl.constants import *

class SequentialModelBASE:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]

    self.input_shape = self.env.observation_space.shape

    # TODO: handle discrete vs continous action_space better
    if hasattr(self.env.action_space, "n"):
      self.n_outputs = self.env.action_space.n
    else:
      self.n_outputs = self.env.action_space.shape[1]

    self.fully_connected_model_size = kwargs.get(FULLY_CONNECTED_MODEL_SIZE) or [100,100]