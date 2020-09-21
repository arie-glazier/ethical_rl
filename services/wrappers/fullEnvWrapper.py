import gym

class fullEnvWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    