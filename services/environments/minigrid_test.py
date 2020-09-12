from gym_minigrid.minigrid import Floor
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.register import register
import statistics

class Arie5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)
        self.constraint_violation_count = 0
        self.episode_action_history = []

    def _gen_grid(self, width, height):
      super()._gen_grid(width, height)

      # Put a yellow square in the middle
      yellow_coords = (width//2, height//2)
      self.metadata["yellow_coords"] = [yellow_coords]
      self.put_obj(Floor("yellow"), *yellow_coords)

    def step(self, action):
      obs, reward, done, info = super().step(action)

      # note constraint violation if step on floor
      if tuple(self.agent_pos) in self.metadata["yellow_coords"]:
        self.constraint_violation_count += 1

      self.episode_action_history.append(action)

      return obs, reward, done, info

    def reset(self):
      self.constraint_violation_count = 0
      self.episode_action_history = []
      return super().reset()

    def _reward(self):
      return 1

    # if we want to play interactively we need to figure this out
    # def get_keys_to_action(self):
    #   return {(): 0, (32,): 1, (100,): 2, (97,): 3, (32, 100): 4, (32, 97): 5}

register(
    id='MiniGrid-arie-test-v0',
    entry_point='services.environments.minigrid_test:Arie5x5'
)