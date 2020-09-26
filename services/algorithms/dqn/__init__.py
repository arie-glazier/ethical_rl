import sys
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from services.algorithms import AlgorithmBASE
from services.constants import *
from services.util import load_replay_buffer
from services.common.schedules import LinearSchedule

class DQNBASE(AlgorithmBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.replay_buffer = load_replay_buffer(kwargs[REPLAY_BUFFER_MODULE])(**kwargs)

    # This is a prioritized replay parameter. TODO: Make configurable
    # This value relects an "amount of prioritization" (starts small -> high).
    # The idea is that training instability in the beginning implies that importance sampling
    # is more important towards the end.
    self.beta_schedule = LinearSchedule(schedule_timesteps=self.number_of_episodes, final_p=1.0, initial_p=0.4) 

    # This is how long we will wait before we start training the model - no reason to train until there's
    # enough data in the buffer.
    self.buffer_wait_steps = int(kwargs[BUFFER_WAIT_STEPS])

  def sample_experiences(self, episode_number):
    states, actions, rewards, next_states, dones, weights, buffer_indexes = self.replay_buffer.sample(self.batch_size, self.beta_schedule.value(episode_number))
    return states, actions, rewards, next_states, dones, weights, buffer_indexes

  def play_one_step(self, state, epsilon):
    action = self.policy.get_action(self.model, state, epsilon)
    next_state, reward, done, info = self.env.step(action)
    self.replay_buffer.add(state, action, reward, next_state, done)
    return next_state, reward, done, info

  def train(self):
    pass