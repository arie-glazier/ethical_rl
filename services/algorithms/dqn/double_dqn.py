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

  def __training_step(self, episode_number):
    states, actions, rewards, next_states, dones, weights, buffer_indexes = self.sample_experiences(episode_number)
    next_Q_values = self.model.predict(next_states)

    # this makes a double DQN
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
    next_best_Q_values = (self.target_model.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards + (1-dones) * self.discount_factor * next_best_Q_values)

    mask = tf.one_hot(actions, self.n_outputs)
    with tf.GradientTape() as tape:
      all_Q_values = self.model(states) # this is a tf 2.0.0 issue, when we upgrade the cast can be removed
      Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
      loss = tf.reduce_mean(weights * self.loss_function(target_Q_values, Q_values))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    # this is TD error (i think)
    # TODO: really think about this so we're sure it is the correct calculation
    td_error = np.abs(np.subtract(Q_values.numpy().flatten(), target_Q_values))
    # TODO: make this configurable
    distribution_shape = 0.5
    weighted_td_error = np.power(td_error, distribution_shape)

    # update priority replay buffer
    self.replay_buffer.update_priorities(buffer_indexes, weighted_td_error)

  def train(self):
    rewards = []
    for episode in range(self.number_of_episodes):
      state = self.env.reset()
      total_episode_rewards = 0
      epsilon = self.epsilon_schedule.value(episode)
      for step in range(self.maximum_step_size):
        state, reward, done, info = self.play_one_step(state, epsilon)
        total_episode_rewards += reward
        if done:
          break

      # copy weights to target every X episodes
      if episode >= self.buffer_wait_steps and episode % self.target_sync_frequency == 0: 
        print(f"completed episode {episode} with reward {total_episode_rewards}")
        self.target_model.set_weights(self.model.get_weights())

      # no need to train until the buffer has data
      if episode >= self.buffer_wait_steps: 
        self.__training_step(episode)

      # print(f"episode: {episode} / total_rewards: {total_episode_rewards} / total_steps: {step} / metadata: {self.env.metadata}")
      # TODO: reward module that captures arbitrary data
      rewards.append(total_episode_rewards)
    return rewards
