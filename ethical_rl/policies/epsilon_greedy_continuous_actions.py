import numpy as np
import sys
from ethical_rl.constants import *
from scipy.spatial import distance
import tensorflow as tf
import pandas as pd

class Policy:
  def __init__(self, **kwargs):
    self.env = kwargs[ENVIRONMENT]
    self.available_actions = kwargs["available_actions"]


  def get_action(self, model, state, epsilon):
    if np.random.rand() < epsilon: 
      return self.available_actions.sample(1).item
    else: 
      state = np.array(state)
      possible_actions = self.available_actions.sample(10, replace=True).item.tolist()
      possible_contexts = np.array([np.mean([state, x], axis=0).tolist() for x in possible_actions])
      predictions = model.predict(possible_contexts)
      reduced_predictions = tf.reduce_sum(predictions, 1, keepdims=True)
      prediction = pd.DataFrame(possible_actions[np.argmax(reduced_predictions)])
      return prediction