import sys
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from services.algorithms import AlgorithmBASE
from services.constants import *

class Algorithm(AlgorithmBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.target_model = keras.models.clone_model(self.model) # fixed Q targets
    self.target_model.set_weights(self.model.get_weights())

    self.replay_buffer = deque(maxlen=kwargs[MAX_REPLAY_BUFFER_LENGTH]) # change to a circular buffer if replay length gets long

    self.buffer_wait_steps = kwargs[BUFFER_WAIT_STEPS]

  def __sample_experiences(self):
    indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
    batch = [ self.replay_buffer[index] for index in indices ]
    states, actions, rewards, next_states, dones = [ np.array([experience[field_index] for experience in batch])for field_index in range(5) ]
    return states, actions, rewards, next_states, dones

  def __play_one_step(self, state, epsilon):
    action = self.policy.get_action(self.model, state, epsilon)
    next_state, reward, done, info = self.env.step(action)
    self.replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

  def __training_step(self):
    states, actions, rewards, next_states, dones = self.__sample_experiences()
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
        state, reward, done, info = self.__play_one_step(state, epsilon)
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
