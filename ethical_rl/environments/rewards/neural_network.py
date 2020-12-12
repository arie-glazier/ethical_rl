from ethical_rl.environments.rewards import RewardBASE
from ethical_rl.constants import *
import tensorflow as tf
from tensorflow import keras
import numpy as np

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.model = keras.models.load_model("./saved_models/reward_model.h5")
    self.termination_reward = int(kwargs.get(TERMINATION_REWARD, 30))

  # TODO: make util and share w/ wrapper
  def __one_hot_map(self, attribute_size, attribute_observation):
    array = np.zeros(attribute_size)
    array[attribute_observation] = 1
    return array

  def get(self, **kwargs):
    if kwargs["done"]: return self.termination_reward
    state = kwargs["state"]
    agent_direction = state["agent_direction"]
    agent_position = state["agent_position"]
    action = kwargs["action"]
    # do one hot encoding
    input_array = np.concatenate(
      (
        self.__one_hot_map(4, agent_direction),
        self.__one_hot_map(3, agent_position[0]-1),
        self.__one_hot_map(3, agent_position[1]-1),
        self.__one_hot_map(3, action)
      )
    )
    return self.model.predict(np.vstack([input_array]))[0][0]
