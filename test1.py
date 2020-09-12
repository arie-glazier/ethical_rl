import safety_gym
import gym
import gym_minigrid
from services.environments.minigrid_test import *

env = gym.make('MiniGrid-arie-test-v0')

obs = env.reset()
# print(obs)
# print(env.agent_dir)
# print(env.action_space.n)

print(dir(env))
print(env.metadata)

for _ in range(0,100):
  env.render()
  action = env.action_space.sample()
  print(f"{_} / {action}")
  env.step(action) # take a random action
print(env.constraint_violation_count)
env.close()