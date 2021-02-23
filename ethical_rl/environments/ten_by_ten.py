from gym_minigrid.minigrid import Floor
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.register import register
from gym import spaces

import statistics, sys
from enum import IntEnum
import numpy as np

from ethical_rl.constants import *
from ethical_rl.util import load_reward

# TODO: make base class for existing empty environments
class Ethical10x10(EmptyEnv):
    class Actions(IntEnum):
          # Turn left, turn right, move forward
          left = 0
          right = 1
          forward = 2

    def __init__(self, **kwargs):
        self.goal_position = None
        super().__init__(size=10)
        # Action enumeration for this environment
        self.actions = Ethical10x10.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.metadata[CONSTRAINT_VIOLATION_COUNT] = 0
        self.metadata[EPISODE_ACTION_HISTORY] = []

        # In this empty environment, the goal is always in the same place
        self.goal_position = (self.width - 2, self.height - 2)

        self.reward_module_name = kwargs.get(REWARD_MODULE, "ethical_rl.environments.rewards.negative_step")
        self.reward_module = load_reward(self.reward_module_name)(environment=self, **kwargs)

        self.max_steps = int(kwargs.get(MAX_STEPS_PER_EPISODE,100))

        if kwargs.get(RANDOM_START_POSITION): self.agent_start_pos = None

    def _gen_grid(self, width, height):
      super()._gen_grid(width, height)

      path_to_take = []
      for i in range(1,width-1):
        for j in range(1,height-1):
          if i == j + 1 or i == j:
            path_to_take.append((i, j))
      
      yellow_coords = [ (i,j) for i in range(1,width-1) for j in range(1,height-1) if (i,j) not in path_to_take ]
      self.metadata[YELLOW_COORDINATES] = yellow_coords

      for position in yellow_coords:  self.put_obj(Floor(YELLOW), *position)

    def _reward(self, done, constraint_violation):
      """
      Compute the reward to be given upon success
      """
      return self.reward_module.get(done, constraint_violation)

    def step(self, action):
      self.step_count += 1

      reward = 0
      done = False
      info = {}

      # Get the position in front of the agent
      fwd_pos = self.front_pos

      # Get the contents of the cell in front of the agent
      fwd_cell = self.grid.get(*fwd_pos)

      # Rotate left
      if action == self.actions.left:
          self.agent_dir -= 1
          if self.agent_dir < 0:
              self.agent_dir += 4

      # Rotate right
      elif action == self.actions.right:
          self.agent_dir = (self.agent_dir + 1) % 4

      # Move forward
      elif action == self.actions.forward:
          if fwd_cell == None or fwd_cell.can_overlap():
              self.agent_pos = fwd_pos
          if fwd_cell != None and fwd_cell.type == 'goal':
              done = True
          if fwd_cell != None and fwd_cell.type == 'lava':
              done = True

      constraint_violation = tuple(self.agent_pos) in self.metadata[YELLOW_COORDINATES]
      reward = self._reward(done, constraint_violation)

      obs = self.gen_obs()

      # note constraint violation
      if constraint_violation:
        self.metadata[CONSTRAINT_VIOLATION_COUNT] += 1

      self.metadata[EPISODE_ACTION_HISTORY].append(action)

      # add info for symbolic representation
      obs = self.__add_symbols(obs)

      return obs, reward, done, info

    def reset(self):
      self.metadata[CONSTRAINT_VIOLATION_COUNT] = 0
      self.metadata[EPISODE_ACTION_HISTORY] = []
      obs = super().reset()
      obs = self.__add_symbols(obs)

      return obs

    def __add_symbols(self, obs):
      obs["agent_position"] = self.agent_pos
      obs["goal_position"] = self.goal_position
      obs["yellow_position"] = self.metadata[YELLOW_COORDINATES]
      obs["constraint_violation_count"] = self.metadata[CONSTRAINT_VIOLATION_COUNT]
      obs["step_count"] = self.step_count
      return obs

register(
    id='MiniGrid-Ethical10x10-v0',
    entry_point='ethical_rl.environments.ten_by_ten:Ethical10x10'
)