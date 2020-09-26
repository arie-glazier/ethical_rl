import gym
from gym.utils.play import play
import safety_gym
import argparse
from services.environments.minigrid_test import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--game_name", default='MiniGrid-arie-test-v0')

if __name__ == "__main__":
  args = PARSER.parse_args()
  env = gym.make(args.game_name)
  env.reset()
  for action in [1, 2, 2, 0, 2, 0, 0, 0, 0, 2]:
    env.render()
    result = env.step(action)
  env.close()
  # play(env)