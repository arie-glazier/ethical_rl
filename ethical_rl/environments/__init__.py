from ethical_rl.constants import *
from ethical_rl.util import load_reward

from gym_minigrid.minigrid import Floor
from gym_minigrid.envs.empty import EmptyEnv

from gym import spaces
from enum import IntEnum
import numpy as np

import sys

class PossibleActions(IntEnum):
  # Turn left, turn right, move forward
  left = 0
  right = 1
  forward = 2

  # Pick up an object
  pickup = 3
  # Drop an object
  drop = 4
  # Toggle/activate an object
  toggle = 5

  # Done completing task
  done = 6

class EthicalGridBASE(EmptyEnv):
  def __init__(self,  **kwargs):
    self.goal_position = None
    self.size = kwargs["size"]
    super().__init__(size=self.size)

    # Action enumeration for this environment
    self.available_actions = self._get_available_actions() 
    self.actions = IntEnum("Actions", self.available_actions)
    self.possible_actions = PossibleActions
    # Actions are discrete integer values
    self.action_space = spaces.Discrete(len(self.actions))

    self.metadata[CONSTRAINT_VIOLATION_COUNT] = 0
    self.metadata[EPISODE_ACTION_HISTORY] = []

    # In this empty environment, the goal is always in the same place
    self.goal_position = (self.width - 2, self.height - 2)

    self.reward_module_name = kwargs.get(REWARD_MODULE, "ethical_rl.environments.rewards.negative_step_constraint_aware")
    self.reward_module = load_reward(self.reward_module_name)(environment=self, **kwargs)

    self.max_steps = int(kwargs.get(MAX_STEPS_PER_EPISODE,20))

    if kwargs.get(RANDOM_START_POSITION): 
      self.agent_start_pos = None

  def _get_available_actions(self):
    return PossibleActions

  def _gen_grid(self, width, height):
    super()._gen_grid(width, height)
    # custom grid code goes here for children

  def _reward(self, **kwargs):
    return self.reward_module.get(**kwargs)

  def _drop(self, action, fwd_cell, fwd_pos):
    if (not fwd_cell or fwd_cell.can_contain()) and self.carrying:
      self.grid.set(*fwd_pos, self.carrying)
      self.carrying.cur_pos = fwd_pos
      self.carrying = None

  def _take_step(self, action):
    self.step_count += 1

    current_state = {
      "agent_position":self.agent_pos,
      "agent_direction":self.agent_dir
      }

    reward = 0
    done = False

    # Get the position in front of the agent
    fwd_pos = self.front_pos

    # Get the contents of the cell in front of the agent
    fwd_cell = self.grid.get(*fwd_pos)

    # Rotate left
    if action == self.possible_actions.left:
        self.agent_dir -= 1
        if self.agent_dir < 0:
            self.agent_dir += 4

    # Rotate right
    elif action == self.possible_actions.right:
        self.agent_dir = (self.agent_dir + 1) % 4

    # Move forward
    elif action == self.possible_actions.forward:
        if fwd_cell == None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos
        if fwd_cell != None and fwd_cell.type == 'goal':
            done = True
        if fwd_cell != None and fwd_cell.type == 'lava':
            done = True
    
    # Pick up an object
    elif action == self.possible_actions.pickup:
        if fwd_cell and fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(*fwd_pos, None)

    # Drop an object
    elif action == self.possible_actions.drop:
        self._drop(action, fwd_cell, fwd_pos)

    # Toggle/activate an object
    elif action == self.possible_actions.toggle:
        if fwd_cell:
            fwd_cell.toggle(self, fwd_pos)

    # Done action (not used by default)
    elif action == self.possible_actions.done:
        pass
    else:
        assert False, "unknown action"


    obs = self.gen_obs()
    return obs, done

  def step(self, action):
    info = {}
    obs, done = self._take_step(action)

    # note constraint violation
    constraint_violation = self._handle_constraint_violation(obs,reward,done,info)

    self.metadata[EPISODE_ACTION_HISTORY].append(action)

    reward = self._reward(done=done, constraint_violation=constraint_violation, action=action, state=current_state)

    return obs, reward, done, info

  def _handle_constraint_violation(self, *args):
    # example: 
    # tuple(self.agent_pos) in self.metadata[YELLOW_COORDINATES] 
    #  if constraint_violation:
    #  self.metadata[CONSTRAINT_VIOLATION_COUNT] += 1
    raise NotImplementedError

  def reset(self):
    self.metadata[CONSTRAINT_VIOLATION_COUNT] = 0
    self.metadata[EPISODE_ACTION_HISTORY] = []
    obs = super().reset()
    obs = self._add_symbols(obs)
    return obs

  def _add_symbols(self, obs):
    obs["agent_position"] = self.agent_pos
    obs["goal_position"] = self.goal_position
    obs["constraint_violation_count"] = self.metadata[CONSTRAINT_VIOLATION_COUNT]
    obs["step_count"] = self.step_count
    return obs