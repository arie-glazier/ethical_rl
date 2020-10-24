import sys
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from services.algorithms.dqn import DQNBASE
from services.constants import *

class Algorithm(DQNBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.target_model = keras.models.clone_model(self.model) # fixed Q targets
    self.target_model.set_weights(self.model.get_weights())
    self.target_sync_frequency = int(kwargs[TARGET_SYNC_FREQUENCY])

  def _get_target_q_values(self, next_Q_values, rewards, dones, next_states, *args):
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
    next_best_Q_values = (self.target_model.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards + (1-dones) * self.discount_factor * next_best_Q_values)
    return target_Q_values

  def _train_single_episode(self, episode):
    reward, loss = super()._train_single_episode(episode)

    # copy weights to target every X episodes
    if episode >= self.buffer_wait_steps and episode % self.target_sync_frequency == 0: 
      print(f"completed episode {episode} with reward {reward}")
      self.target_model.set_weights(self.model.get_weights())
    
    return reward, loss