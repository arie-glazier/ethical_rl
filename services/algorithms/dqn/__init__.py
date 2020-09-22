import sys
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from services.algorithms import AlgorithmBASE
from services.constants import *

class DQNBASE(AlgorithmBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.replay_buffer = deque(maxlen=kwargs[MAX_REPLAY_BUFFER_LENGTH]) # change to a circular buffer if replay length gets long

    self.buffer_wait_steps = kwargs[BUFFER_WAIT_STEPS]

  def sample_experiences(self):
    indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
    batch = [ self.replay_buffer[index] for index in indices ]
    states, actions, rewards, next_states, dones = [ np.array([experience[field_index] for experience in batch])for field_index in range(5) ]
    return states, actions, rewards, next_states, dones

  def play_one_step(self, state, epsilon):
    action = self.policy.get_action(self.model, state, epsilon)
    next_state, reward, done, info = self.env.step(action)
    self.replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


  def train(self):
    pass