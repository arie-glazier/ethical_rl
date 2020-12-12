from ethical_rl.environments.rewards import RewardBASE
from ethical_rl.constants import *

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.termination_reward = int(kwargs.get(TERMINATION_REWARD, 30))
    self.step_reward = int(kwargs.get(STEP_REWARD, -1))
    self.constraint_violation_penalty = int(kwargs.get("constraint_violation_penalty", -1))

  def get(self, **kwargs):
    if kwargs["done"]:
      return self.termination_reward
    return self.step_reward + (self.constraint_violation_penalty if kwargs.get("constraint_violation") else 0)