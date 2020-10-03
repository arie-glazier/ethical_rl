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
    # TODO: WHAT IS UP W RESHAPE????
    target_Q_values = (rewards + (1-dones) * self.discount_factor * next_best_Q_values)#.reshape(-1,1)
    return target_Q_values

  def train(self):
    rewards = []
    losses = []
    for episode in range(self.number_of_episodes):
      state = self.env.reset()
      total_episode_rewards = 0
      epsilon = self.epsilon_schedule.value(episode - self.buffer_wait_steps) if episode >= self.buffer_wait_steps else 1.0
      for step in range(self.maximum_step_size):
        state, reward, done, info = self.play_one_step(state, epsilon)
        total_episode_rewards += reward
        if done:
          break

      # no need to train until the buffer has data
      if episode >= self.buffer_wait_steps: 
        loss = self._training_step(episode)
        losses.append(loss)
      else:
        losses.append(0)

      # copy weights to target every X episodes
      if episode >= self.buffer_wait_steps and episode % self.target_sync_frequency == 0: 
        print(f"completed episode {episode} with reward {total_episode_rewards}")
        self.target_model.set_weights(self.model.get_weights())

      print(f"episode: {episode} / total_rewards: {total_episode_rewards} / total_steps: {step} / epsilon: {epsilon}")
      # TODO: reward module that captures arbitrary data
      rewards.append(total_episode_rewards)
      if rewards[-10:] == [-100] * 10: break
    return rewards, losses
