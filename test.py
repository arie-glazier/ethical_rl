from ethical_rl.environments.five_by_five_object import *
from ethical_rl.util import load_class, load_object, load_model, load_policy, load_algorithm

import gym, time

name = 'MiniGrid-Ethical5x5Object-v0'
environment_wrapper_config = {
      "modules" : ["ethical_rl.wrappers.symbolic_observations"], 
      "classes" : ["SymbolicObservationsOneHotWrapperObject"]
    }
env = gym.make(name, reward_module="ethical_rl.environments.rewards.object_constraint_aware")
if environment_wrapper_config:
  for idx, module in enumerate(environment_wrapper_config[MODULES]): # for ability to wrap mulitple
    environment_wrapper = load_class(module, environment_wrapper_config[CLASSES][idx])
    env = environment_wrapper(env)
result = env.reset()

optimal_path = [1, 2, 3, 0, 2, 2, 0, 4, 1, 1, 2]
path = [1,2,3,2,4]
path = [1, 2, 3, 2, 0, 2, 2]
path = [1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
path = [1, 2, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
path = [1, 2, 3, 0, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3]
path = [1, 2, 3, 0, 2, 0, 2, 1, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3]
path = [1, 2, 3, 0, 2, 2, 2, 2, 2, 0, 4, 0, 0, 2]
path = [3, 2, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]
path = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
path = [3, 2, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
path = [2, 3, 2, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
path = [1, 2, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
path = [1, 2, 3, 0, 2, 2, 0, 4, 1, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
path = [3, 2, 4, 1, 2, 2, 0, 2]

for action in optimal_path:
  time.sleep(.5)
  result = env.step(action)
  print(f"{action}:{result}")
  # print(dir(env.carrying))
  env.render()

# input("waiting")
env.close()