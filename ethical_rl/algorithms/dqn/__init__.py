import sys
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ethical_rl.algorithms import AlgorithmBASE
from ethical_rl.constants import *
from ethical_rl.util import load_replay_buffer
from ethical_rl.common.schedules.linear import Schedule as LinearSchedule

class DQNBASE(AlgorithmBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.m = -10000
    self.history = defaultdict(list)

    self.replay_buffer = load_replay_buffer(kwargs[REPLAY_BUFFER_MODULE])(**kwargs)

    # This is a prioritized replay parameter. TODO: Make configurable
    # This value relects an "amount of prioritization" (starts small -> high).
    # The idea is that training instability in the beginning implies that importance sampling
    # is more important towards the end.
    self.beta_schedule = LinearSchedule(schedule_timesteps=self.number_of_episodes, final_p=1.0, initial_p=0.4) 

    # This is how long we will wait before we start training the model - no reason to train until there's
    # enough data in the buffer.
    self.buffer_wait_steps = int(kwargs[BUFFER_WAIT_STEPS])

  def _sample_experiences(self, episode_number):
    states, actions, rewards, next_states, dones, weights, buffer_indexes = self.replay_buffer.sample(self.batch_size, self.beta_schedule.value(episode_number))
    return states, actions, rewards, next_states, dones, weights, buffer_indexes

  def _play_one_step(self, state, epsilon):
    action = self.policy.get_action(self.model, state, epsilon)
    next_state, reward, done, info = self.env.step(action)
    self.replay_buffer.add(state, action, reward, next_state, done)
    return next_state, reward, done, info

  def _update_model(self, states, actions, weights, target_Q_values):
    # TODO: Understand exactly what GradientTape does.
    mask = tf.one_hot(actions, self.n_outputs)
    with tf.GradientTape() as tape:
      all_Q_values = self.model(states)
      Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
      # TODO: check that multiplication of weights here is correct.
      loss = tf.reduce_mean(weights * self.loss_function(target_Q_values.reshape(-1,1), Q_values))
    grads = tape.gradient(loss, self.model.trainable_variables)

    # Apply clipping by global norm
    # TODO: see if norm clipping can just be incorporated into the Adam optimizer
    if self.clip_norm:
      grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)

    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    return Q_values, loss

  def _update_replay_buffer(self, Q_values, target_Q_values, buffer_indexes):
    # this is TD error (i think)
    # TODO: really think about this so we're sure it is the correct calculation
    td_error = np.abs(np.subtract(Q_values.numpy().flatten(), target_Q_values.flatten()))
    # TODO: make this configurable
    distribution_shape = 0.5
    weighted_td_error = np.power(td_error, distribution_shape).reshape(-1)
    # print(f"weighted error: {weighted_td_error}")

    # update priority replay buffer
    try:
      self.replay_buffer.update_priorities(buffer_indexes, weighted_td_error)
    except Exception as ex:
      print(f"Buffer Exception: {ex}")
      print(weighted_td_error)

  def _training_step(self, episode_number):
    states, actions, rewards, next_states, dones, weights, buffer_indexes = self._sample_experiences(episode_number)
    next_Q_values = self.model.predict(next_states)

    target_Q_values = self._get_target_q_values(next_Q_values, rewards, dones, next_states)
    Q_values, loss = self._update_model(states, actions, weights, target_Q_values)
    self._update_replay_buffer(Q_values, target_Q_values, buffer_indexes)

    return loss

  def _train_single_episode(self, episode):
    state = self.env.reset()
    total_episode_rewards = 0
    epsilon = self.epsilon_schedule.value(episode - self.buffer_wait_steps) if episode >= self.buffer_wait_steps else 1.0
    loss = 0
    for step in range(self.maximum_step_size):
      if self.render_training_steps and episode % self.render_training_steps == 0:
        self.env.render()
      state, reward, done, info = self._play_one_step(state, epsilon)
      if done and self.render_training_steps and episode % self.render_training_steps == 0:
        self.env.step(2)
        self.env.render()
      total_episode_rewards += reward

      # + quit on actually reaching goal or passing the step limit
      # + hasattr to make compatible with other environments
      if done or (hasattr(self.env,"step_count") and self.env.step_count >= self.env.max_steps):
        break

      # no need to train until the buffer has data
      if step % 4 == 0:
        loss = self._training_step(episode) if episode >= self.buffer_wait_steps else 0

    print(f"{episode} / {total_episode_rewards} / {step} / {epsilon} / {self.m} / {np.mean(self.history[REWARDS][-10:])}")

    # TODO: this is dependent on being a minigrid environment
    if hasattr(self.env.env,"agent_start_pos"): print(f"{self.env.env.agent_start_pos} / {self.env.env.agent_start_dir} episode: {episode} / total_rewards: {total_episode_rewards} / total_steps: {step} / epsilon: {epsilon}")

    # TODO: result module that captures arbitrary data
    return total_episode_rewards, loss

  def train(self):
    history = self.history 

    for episode in range(self.number_of_episodes):
      reward, loss = self._train_single_episode(episode)
      history[REWARDS].append(reward)
      history[LOSS].append(loss)
      if reward > self.m: self.m = reward
      
      # TODO: this is dependent on being a minigrid environment
      if hasattr(self.env, "metadata") and CONSTRAINT_VIOLATION_COUNT in self.env.metadata:
        history[CONSTRAINT_VIOLATION_COUNT].append(self.env.metadata[CONSTRAINT_VIOLATION_COUNT])
        history[EPISODE_ACTION_HISTORY].append(self.env.metadata[EPISODE_ACTION_HISTORY])

      # TODO: this is bad, can do better (wrapper is weird too)
      # TODO: this is dependent on being a minigrid environment
      # if hasattr(self.env.metadata, EPISODE_ACTION_HISTORY) and self.number_of_episodes - episode <= 200: 
      if self.number_of_episodes - episode <= 200: 
        print(self.env.metadata[EPISODE_ACTION_HISTORY])
        self.env.env.agent_start_pos = (1,1)
        self.env.env.random_ball_position = False
        # self.render_training_steps = 10

    return history

  def _get_target_q_values(self, *args):
    raise NotImplementedError("Implemented By Child")
