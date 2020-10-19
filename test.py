import json, argparse, os, sys, importlib

import gym
from services.environments.minigrid_test import *
from services.util import load_class, load_object, load_model, load_policy, load_algorithm
from services.arguments import Arguments
from services.config import Config
from services.constants import *
import multiprocessing
import time
import threading
from threading import Thread, Event
global exit
exit = Event()

f = open('config.json',) # Opening config file 
tests = json.load(f).keys() # getting test names
f.close() # Closing config file 

pwd = os.path.dirname(os.path.realpath(__file__))

for test in tests: # Looping over each test

  # Mostly copied from main.py
  PARSER = Arguments(pwd=pwd).parser 
  PARSER.add_argument("--test_name")
  if __name__ == "__main__":
    args = PARSER.parse_args()
    setattr(args, '--test_name', test) # Add test name as argument
    print('\n',test,'\n')
    config = Config(pwd=pwd, args=args).config
    env = gym.make(config.get(ENVIRONMENT_NAME, args.environment_name), **config)
    environment_wrapper_config = config.get(ENVIRONTMENT_WRAPPER)
    if environment_wrapper_config:
      for idx, module in enumerate(environment_wrapper_config[MODULES]): # for ability to wrap mulitple
        environment_wrapper = load_class(module, environment_wrapper_config[CLASSES][idx])
        env = environment_wrapper(env)
    env.reset()
    model = load_model(config[MODEL_MODULE])(environment=env, **config).model
    policy = load_policy(config[POLICY_MODULE])(environment=env)
    algorithm = load_algorithm(config[ALGORITHM_MODULE])(environment=env, model=model, policy=policy, **config)
    algorithm.train(timed=True) # Run each test for 5 episodes