import time, sys
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
    self.evaluate_steps = int(kwargs["evaluate_steps"])
    self.render_steps = kwargs["render_steps"]
    self.loss_steps = kwargs["loss_steps"]
    self.max_steps_per_episode = int(kwargs[MAX_STEPS_PER_EPISODE])
    self.rollout_length = int(kwargs["rollout_length"])
    self.num_epochs = int(kwargs["num_epochs"])

    self.value_net = self.model
    self.actor_net = self.policy.model

    self.global_step = tf.compat.v1.train.get_or_create_global_step()

    self.tf_agent = PPOAgent(
        self.env.time_step_spec(),
        self.env.action_spec(),
        self.optimizer,
        self.actor_net,
        self.value_net,
        num_epochs=self.num_epochs,
        train_step_counter=self.global_step,
        discount_factor=kwargs[DISCOUNT_FACTOR],
        gradient_clipping=kwargs[CLIP_NORM],
        importance_ratio_clipping=kwargs["clip_ratio"],
        use_gae=True,
        normalize_rewards=False,
        lambda_value = kwargs["td_lambda_value"],
        use_td_lambda_return=True,
        adaptive_kl_target=kwargs["target_kl"],
        normalize_observations=False,
        debug_summaries=True,
    )
    self.tf_agent.initialize()

    self.replay_buffer = load_replay_buffer(kwargs[REPLAY_BUFFER_MODULE])(tf_agent=self.tf_agent, **kwargs).replay_buffer

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
    policy._validate_args = False
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        step_counter = 0
        episode_actions = []
        while not time_step.is_last() and step_counter <= self.max_steps_per_episode:
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
    all_experience = []
    for _ in range(self.number_of_episodes):
      self.env.reset()
      for _ in range(self.rollout_length):
        self._collect_step(self.env, self.tf_agent.collect_policy, self.replay_buffer)

      experience, unused_info = next(self.iterator)
      all_experience.append(experience)
      train_loss = self.tf_agent.train(experience).loss
      self.replay_buffer.clear()
      step = self.tf_agent.train_step_counter.numpy()

      # Print loss 
      if self.loss_steps and step % self.loss_steps == 0:
          print('step = {0}: loss = {1}'.format(step, train_loss))
      # Evaluate agent's performance 
      if self.evaluate_steps and step % self.evaluate_steps == 0:
          avg_return = self._compute_avg_return(self.env, self.tf_agent.policy, 5)
          print('step = {0}: Average Return = {1}'.format(step, avg_return))
          returns.append(avg_return)
          # proceed = input("proceed? ")
          # if proceed == "n":
          #   break
    print(returns)
    if self.render_steps and self.evaluate_steps: self._plot_results(returns)
    return all_experience
  
  def _plot_results(self, returns):
    iterations = range(0, self.number_of_episodes + 1, self.evaluate_steps)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations') 
    plt.show()

    for test_no in range(5):
      time_step = self.env.reset()
      self.env._env.envs[0].gym.render()
      r = 0
      for step in range(self.render_steps):
        action = self.tf_agent.collect_policy.action(time_step)
        time_step = self.env.step(action.action)
        self.env._env.envs[0].gym.render()
        r += time_step.reward
        time.sleep(0.1)
      print(f"reward: {r}")