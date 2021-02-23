import gym, time, sys
import safety_gym
import argparse
from ethical_rl.environments.minigrid_test import *
from ethical_rl.environments.five_by_five import *
from ethical_rl.environments.ten_by_ten import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--game_name", default='MiniGrid-Ethical10x10-v0')

if __name__ == "__main__":
  args = PARSER.parse_args()
  env = gym.make(args.game_name)
  env.reset()
  env.render()
  for action in [2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2]:
    time.sleep(.5)
    result = env.step(action)
    env.render()
  time.sleep(.5)
  env.close()