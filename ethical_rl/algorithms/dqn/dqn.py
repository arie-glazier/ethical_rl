import sys
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ethical_rl.algorithms.dqn import DQNBASE
from ethical_rl.constants import *

class  Algorithm(DQNBASE):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)

    def _get_target_q_values(self, next_Q_values, rewards, dones, *args):
      max_next_Q_values = np.max(next_Q_values, axis=1)
      target_Q_values = (rewards +
                          (1 - dones) * self.discount_factor * max_next_Q_values)
      target_Q_values = target_Q_values.reshape(-1, 1)
      return target_Q_values