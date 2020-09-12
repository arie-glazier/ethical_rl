import sys
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras

class DQN:
  def __init__(self, **kwargs):
    self.env = kwargs["environment"]
    self.n_outputs = self.env.action_space.n
    self.model = kwargs["model"]
    self.policy = kwargs["policy"]

    self.replay_buffer = deque(maxlen=kwargs["max_replay_buffer_length"]) # change to a circular buffer if replay length gets long
    self.batch_size = kwargs["batch_size"]
    self.discount_factor = kwargs["discount_factor"]
    self.loss_function = getattr(keras.losses, kwargs["loss_function"])
    self.optimizer = getattr(keras.optimizers, kwargs["optimizer"])(lr=kwargs["learning_rate"])

    self.number_of_episodes = kwargs["number_of_episodes"]
    self.maximum_step_size = kwargs["maximum_step_size"]
    self.buffer_wait_steps = kwargs["buffer_wait_steps"]

  def __sample_experiences(self):
    indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
    batch = [ self.replay_buffer[index] for index in indices ]
    states, actions, rewards, next_states, dones = [ np.array([experience[field_index] for experience in batch])for field_index in range(5) ]
    return states, actions, rewards, next_states, dones

  def __play_one_step(self, state, epsilon):
    action = self.policy.get_action(state, epsilon)
    next_state, reward, done, info = self.env.step(action)
    self.replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

  def __training_step(self):
    states, actions, rewards, next_states, dones = self.__sample_experiences()
    next_Q_values = self.model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis = 1)
    target_Q_values = (rewards + (1-dones) * self.discount_factor * max_next_Q_values).reshape(-1,1)
    mask = tf.one_hot(actions, self.n_outputs)
    with tf.GradientTape() as tape:
      all_Q_values = self.model(states)
      Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
      loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

  def train(self):
    for episode in range(self.number_of_episodes):
      state = self.env.reset()
      total_episode_rewards = 0
      for step in range(self.maximum_step_size):
        epsilon = max(1 - episode / 500, 0.01) # this can be configurable
        state, reward, done, info = self.__play_one_step(state, epsilon)
        total_episode_rewards += reward
        if done:
          break
      if episode > self.buffer_wait_steps: # no need to train until the buffer has data
        self.__training_step()

      print(f"episode: {episode} / total_rewards: {total_episode_rewards} / constraint_violations: {self.env.constraint_violation_count}")
