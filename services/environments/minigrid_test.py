from gym_minigrid.minigrid import Floor
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.register import register
from gym import spaces

import statistics, sys
import numpy
from enum import IntEnum
from services.constants import *

class Arie5x5(EmptyEnv):
    class Actions(IntEnum):
          # Turn left, turn right, move forward
          left = 0
          right = 1
          forward = 2

    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)
        # Action enumeration for this environment
        self.actions = Arie5x5.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.metadata[CONSTRAINT_VIOLATION_COUNT] = 0
        self.metadata[EPISODE_ACTION_HISTORY] = []

    def _gen_grid(self, width, height):
      super()._gen_grid(width, height)
      # Put a yellow square in the middle
      yellow_coords = (width//2, height//2)
      self.goal_cord = numpy.asarray((self.width-2,self.height-2)) #place goal coordinates in the bottom right corner
      self.metadata[YELLOW_COORDINATES] = [yellow_coords]
      self.put_obj(Floor(YELLOW), *yellow_coords)

    def step(self, action):
      obs, reward, done, info = super().step(action)
      
      # note constraint violation if step on floor
      if tuple(self.agent_pos) in self.metadata[YELLOW_COORDINATES]:
        self.metadata[CONSTRAINT_VIOLATION_COUNT] += 1
      self.metadata[EPISODE_ACTION_HISTORY].append(action)
      if self.agent_pos != type(super().place_agent()): self.agent_pos = numpy.asarray(self.agent_pos) # Yes this is sometimes the case
      new_obs = {'player_cord': self.agent_pos,'direction': obs['direction'], 
                  'yellow_cord': [numpy.asarray(sq) for sq in self.metadata[YELLOW_COORDINATES]], 'goal_cord': self.goal_cord, 'violations': self.metadata[CONSTRAINT_VIOLATION_COUNT]}
      print(new_obs)
      #print(obs_list)
      return new_obs, reward, done, info #will run if you return obs

    def reset(self):
      self.metadata[CONSTRAINT_VIOLATION_COUNT] = 0
      self.metadata[EPISODE_ACTION_HISTORY] = []
      return super().reset()

    # def _reward(self):
    #   return 1

    # if we want to play interactively we need to figure this out
    # def get_keys_to_action(self):
    #   return {(): 0, (32,): 1, (100,): 2, (97,): 3, (32, 100): 4, (32, 97): 5}

register(
    id='MiniGrid-arie-test-v0',
    entry_point='services.environments.minigrid_test_fullEnv:Arie5x5'
)