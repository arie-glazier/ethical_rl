import importlib, sys
from ethical_rl.constants import *

def load_replay_buffer(module, class_name="ReplayBuffer"):
  return load_object({MODULE:module, CLASS:class_name})

def load_reward(module, class_name="Reward"):
  return load_object({MODULE:module, CLASS:class_name})

def load_model(module, class_name="Model"):
  return load_object({MODULE:module, CLASS:class_name})

def load_policy(module, class_name="Policy"):
  return load_object({MODULE:module, CLASS:class_name})

def load_algorithm(module, class_name="Algorithm"):
  return load_object({MODULE:module, CLASS:class_name})

def load_object(attrs):
  return load_class(attrs[MODULE], attrs[CLASS])

def load_class(module, class_name):
  print(module)
  return getattr(importlib.import_module(module), class_name)

def load_schedule(module, class_name="Schedule"):
  return load_object({MODULE:module, CLASS:class_name})