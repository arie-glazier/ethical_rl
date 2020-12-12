from gym_minigrid.minigrid import Floor
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.register import register
from gym import spaces

import statistics, sys
from enum import IntEnum
import numpy as np

from ethical_rl.constants import *
from ethical_rl.util import load_reward

class Arie5x5(EmptyEnv):
    class Actions(IntEnum):
          # Turn left, turn right, move forward
          left = 0
          right = 1
          forward = 2

    def __init__(self, **kwargs):
        self.goal_position = None
        super().__init__(size=5)
        # Action enumeration for this environment
        self.actions = Arie5x5.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.metadata[CONSTRAINT_VIOLATION_COUNT] = 0
        self.metadata[EPISODE_ACTION_HISTORY] = []

        # In this empty environment, the goal is always in the same place
        self.goal_position = (self.width - 2, self.height - 2)

        self.reward_module_name = kwargs.get(REWARD_MODULE, "environments.rewards.negative_step")
        self.reward_module = load_reward(self.reward_module_name)(environment=self, **kwargs)

        self.max_steps = int(kwargs[MAX_STEPS_PER_EPISODE])

    def _gen_grid(self, width, height):
      super()._gen_grid(width, height)

      # Put a yellow square in the middle
      yellow_coords = (width//2, height//2)
      self.metadata[YELLOW_COORDINATES] = [yellow_coords]
      self.put_obj(Floor(YELLOW), *yellow_coords)

    def _reward(self, done):
      """
      Compute the reward to be given upon success
      """
      return self.reward_module.get(done)

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

      reward = self._reward(done)

      obs = self.gen_obs()

      # note constraint violation
      if tuple(self.agent_pos) in self.metadata[YELLOW_COORDINATES]:
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

    # if we want to play interactively we need to figure this out
    # def get_keys_to_action(self):
    #   return {(): 0, (32,): 1, (100,): 2, (97,): 3, (32, 100): 4, (32, 97): 5}

register(
    id='MiniGrid-arie-test-v0',
    entry_point='ethical_rl.environments.minigrid_test:Arie5x5'
)