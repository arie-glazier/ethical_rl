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

  def __training_step(self):
    states, actions, rewards, next_states, dones = self.sample_experiences()
    next_Q_values = self.model.predict(next_states)

    # this makes a double DQN
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
    next_best_Q_values = (self.target_model.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards + (1-dones) * self.discount_factor * next_best_Q_values)

    # max_next_Q_values = np.max(next_Q_values, axis = 1)
    # target_Q_values = (rewards + (1-dones) * self.discount_factor * max_next_Q_values).reshape(-1,1)
    mask = tf.one_hot(actions, self.n_outputs)
    with tf.GradientTape() as tape:
      # all_Q_values = self.model(states.astype(np.float32)) # this is a tf 2.0.0 issue, when we upgrade the cast can be removed
      all_Q_values = self.model(states) # this is a tf 2.0.0 issue, when we upgrade the cast can be removed
      Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
      loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

  def train(self):
    rewards = []
    for episode in range(self.number_of_episodes):
      state = self.env.reset()
      total_episode_rewards = 0
      for step in range(self.maximum_step_size):
        epsilon = max(1 - episode / 500, 0.01) # this can be configurable
        state, reward, done, info = self.play_one_step(state, epsilon)
        total_episode_rewards += reward
        if done:
          break

      if episode % 50 == 0: # this can be configurable.  copy weights to target every 50 episodes
        print(f"completed episode {episode} with reward {total_episode_rewards}")
        self.target_model.set_weights(self.model.get_weights())
      if episode > self.buffer_wait_steps: # no need to train until the buffer has data
        self.__training_step()

      # print(f"episode: {episode} / total_rewards: {total_episode_rewards} / total_steps: {step} / metadata: {self.env.metadata}")
      if episode >= self.number_of_episodes - 10:
        rewards.append(total_episode_rewards)
    return rewards
