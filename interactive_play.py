import gym
from gym.utils.play import play
import safety_gym
import argparse
from services.environments.minigrid_test import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--game_name")

if __name__ == "__main__":
  args = PARSER.parse_args()
  env = gym.make('MiniGrid-arie-test-v0')
  play(env)