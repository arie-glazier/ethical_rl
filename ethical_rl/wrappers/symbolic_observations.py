import gym, sys
from gym.spaces import Box, MultiDiscrete, Discrete
import numpy as np

# support for using agent direction and position only
class SymbolicObservationsOneHotWrapper(gym.core.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.n_directions = 4
    self.grid_buffer_size = 2 # a 5x5 grid is really 3x3
    self.x_position_size = self.width - self.grid_buffer_size
    self.y_position_size = self.height - self.grid_buffer_size
    self.array_size = self.n_directions + self.x_position_size + self.y_position_size

    # This sets bounds for each value a state observation can take
    low = np.zeros(self.array_size)
    low[-1] = 0
    high = np.ones(self.array_size)
    high[-1] = 1

    # TODO: observation type should be configurable
    # dims = np.array([[2] * (self.n_directions+self.x_position_size+self.y_position_size)], dtype=np.int64)
    # self.observation_space = MultiDiscrete(dims)
    # self.observation_space = Discrete(2)
    self.observation_space = Box(
      low=low,
      high= high
    )

  def _one_hot_map(self, attribute_size, attribute_observation):
    array = np.zeros(attribute_size)
    array[attribute_observation] = 1
    return array

  def observation(self, obs):
    state = np.concatenate(
      (
        self._one_hot_map(self.n_directions, obs["direction"]),
        self._one_hot_map(self.x_position_size, obs["agent_position"][0] - 1),
        self._one_hot_map(self.y_position_size, obs["agent_position"][1] - 1)
      )
    ).astype(np.int64)
    return state

class SymbolicObservationsOneHotWrapperObject(SymbolicObservationsOneHotWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.array_size = self.array_size + self.x_position_size + self.y_position_size + 1# for ball coords

    # This sets bounds for each value a state observation can take
    low = np.zeros(self.array_size)
    low[-1] = -100
    high = np.ones(self.array_size)
    high[-1] = 1000
    self.observation_space = Box(
      low=low,
      high= high
    )

  def observation(self, obs):
    bpc = obs["ball_placed_correctly"]
    obs = np.concatenate(
      (
        self._one_hot_map(self.n_directions, obs["direction"]),
        self._one_hot_map(self.x_position_size, obs["agent_position"][0] - 1),
        self._one_hot_map(self.y_position_size, obs["agent_position"][1] - 1),
        self._one_hot_map(self.x_position_size, obs["ball_coords"][0] - 1),
        self._one_hot_map(self.y_position_size, obs["ball_coords"][1] - 1)
      )
    )
    # obs = super().observation(obs)
    # object_goal = self.grid.get(*self.metadata["square_coords"][0])
    # ball_in_destination = object_goal.type == "ball" # ball is in its destination
    # obs = np.append(obs, ball_in_destination) 
    obs = np.append(obs, 1 if bpc else 0) 

    return obs
    
