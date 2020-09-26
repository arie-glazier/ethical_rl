import sys
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from services.algorithms.dqn import DQNBASE
from services.constants import *

class  Algorithm(DQNBASE):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)

    def __training_step(self, episode_number):
      states, actions, rewards, next_states, dones, weights, buffer_indexes = self.sample_experiences(episode_number)
      next_Q_values = self.model.predict(next_states)

      # Eli - this and the corresponding block in double_dqn
      # needs to be the different (i.e at class level).  
      # Each needs a method _get_target_q_values(self, next_q_values, rewards, dones)
      max_next_Q_values = np.max(next_Q_values, axis=1)
      target_Q_values = (rewards +
                          (1 - dones) * self.discount_factor * max_next_Q_values)
      target_Q_values = target_Q_values.reshape(-1, 1)

      # TODO: Understand exactly what GradientTape does.
      mask = tf.one_hot(actions, self.n_outputs)
      with tf.GradientTape() as tape:
        all_Q_values = self.model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        # TODO: check that multiplication of weights here is correct.
        loss = tf.reduce_mean(weights * self.loss_function(target_Q_values, Q_values))
      grads = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

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
            epsilon = max(1 - episode / 500, 0.01)  # this can be configurable
            for step in range(self.maximum_step_size):
                state, reward, done, info = self.play_one_step(state, epsilon)
                total_episode_rewards += reward
                if done:
                    break

            if episode >= self.buffer_wait_steps:  # no need to train until the buffer has data
                self.__training_step(episode)

            # print(f"episode: {episode} / total_rewards: {total_episode_rewards} / total_steps: {step} / metadata: {self.env.metadata}")
            rewards.append(total_episode_rewards)
        return rewards