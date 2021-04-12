import time
from ethical_rl.constants import *
from ethical_rl.algorithms import AlgorithmBASE
import tensorflow as tf
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
import matplotlib.pyplot as plt
from ethical_rl.util import load_replay_buffer

class Algorithm(AlgorithmBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.value_net = self.model
    self.actor_net = self.policy.model

    self.global_step = tf.compat.v1.train.get_or_create_global_step()

    # TODO: check if BASE optimizer works
    self.optimizer = tf.compat.v1.train.AdamOptimizer(
      learning_rate=kwargs[LEARNING_RATE], 
      epsilon=1e-5
    )
    self.tf_agent = PPOAgent(
        self.env.time_step_spec(),
        self.env.action_spec(),
        self.optimizer,
        self.actor_net,
        self.value_net,
        num_epochs=1,
        train_step_counter=self.global_step,
        discount_factor=0.995,
        gradient_clipping=0.5,
        entropy_regularization=1e-2,
        importance_ratio_clipping=0.13,
        use_gae=True,
        use_td_lambda_return=True,
        adaptive_kl_target=0.01
    )
    self.tf_agent.initialize()

    self.replay_buffer = load_replay_buffer(kwargs[REPLAY_BUFFER_MODULE])(**kwargs)
    self.replay_buffer = TFUniformReplayBuffer(
      self.tf_agent.collect_data_spec, 
      batch_size=1, 
      max_length=1000000
    )
    self.dataset = self.replay_buffer.as_dataset(
      num_parallel_calls=1, 
      sample_batch_size=1, 
      num_steps=self.batch_size).prefetch(1)
    self.iterator = iter(self.dataset)

  def _collect_step(self, env, policy, buffer):
    time_step = env.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

  def _compute_avg_return(self, environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        step_counter = 0
        episode_actions = []
        while not time_step.is_last() and step_counter <= 10:
            step_counter += 1
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_actions.append(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

  def train(self):
    print("starting")
    returns = [self._compute_avg_return(self.env, self.tf_agent.policy, 5)]
    for _ in range(self.number_of_episodes):
      self.env.reset()
      for _ in range(self.batch_size):
        self._collect_step(self.env, self.tf_agent.collect_policy, self.replay_buffer)

      experience, unused_info = next(self.iterator)
      train_loss = self.tf_agent.train(experience).loss
      self.replay_buffer.clear()
      step = self.tf_agent.train_step_counter.numpy()

      # Print loss every 200 steps.
      # if step % 200 == 0:
      #     print('step = {0}: loss = {1}'.format(step, train_loss))
      # Evaluate agent's performance every 1000 steps.
      if step % 25 == 0:
          avg_return = self._compute_avg_return(self.env, self.tf_agent.policy, 5)
          print('step = {0}: Average Return = {1}'.format(step, avg_return))
          returns.append(avg_return)
    print(returns)
    self._plot_results(returns)
    return None
  
  def _plot_results(self, returns):
    iterations = range(0, self.number_of_episodes + 1, 25)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations') 
    plt.show()

    for test_no in range(5):
      time_step = self.env.reset()
      self.env.render()
      r = 0
      for step in range(5):
        action = self.tf_agent.collect_policy.action(time_step)
        time_step = self.env.step(action.action)
        self.env._env.envs[0].gym.render()
        r += time_step.reward
        time.sleep(0.1)
      print(f"reward: {r}")