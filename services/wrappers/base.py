import gym

class BaseWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)