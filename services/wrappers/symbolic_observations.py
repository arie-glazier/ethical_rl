import gym, sys
from gym.spaces import Box
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
    # TODO: Be better about step count
    low = np.zeros(self.array_size + 1)
    low[-1] = -100
    high = np.ones(self.array_size+1)
    high[-1] = 1000
    self.observation_space = Box(
      low=low,
      high= high
    )

  def __one_hot_map(self, attribute_size, attribute_observation):
    array = np.zeros(attribute_size)
    array[attribute_observation] = 1
    return array

  def observation(self, obs):
    return np.concatenate(
      (
        self.__one_hot_map(self.n_directions, obs["direction"]),
        self.__one_hot_map(self.x_position_size, obs["agent_position"][0] - 1),
        self.__one_hot_map(self.y_position_size, obs["agent_position"][1] - 1),
        np.array([obs["step_count"]])
      )
    )