import numpy as np
import sys
from ethical_rl.constants import *

class Policy:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]
    self.n_outputs = self.env.action_space.n

  def get_action(self, model, state, epsilon):
    return np.random.randint(self.n_outputs) if np.random.rand() < epsilon else np.argmax(model.predict(state[np.newaxis])[0])