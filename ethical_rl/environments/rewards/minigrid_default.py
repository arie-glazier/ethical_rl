from ethical_rl.environments.rewards import RewardBASE

class Reward(RewardBASE):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def get(self, *args):
    #TODO: This doesn't work
    return 1 - 0.9 * (self.environment.step_count / self.environment.max_steps)