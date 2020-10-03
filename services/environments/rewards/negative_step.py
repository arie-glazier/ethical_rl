from services.environments.rewards import RewardBASE
from services.constants import *

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.termination_reward = kwargs.get(TERMINATION_REWARD, 30)
    self.step_reward = kwargs.get(STEP_REWARD, -1)

  def get(self, done, *args):
    # TODO: Check for goal state itself
    if done and self.environment.agent_pos[0] == 3 and self.environment.agent_pos[1] == 3:
      return self.termination_reward
    return self.step_reward
