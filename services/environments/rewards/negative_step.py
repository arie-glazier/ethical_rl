from services.environments.rewards import RewardBASE
from services.constants import *

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.termination_reward = int(kwargs.get(TERMINATION_REWARD, 30))
    self.step_reward = int(kwargs.get(STEP_REWARD, -1))

  def get(self, done, *args):
    if done:
      return self.termination_reward
    return self.step_reward
