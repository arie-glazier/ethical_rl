import importlib
from services.constants import *

def load_model(module, class_name="Model"):
  return load_object({MODULE:module, CLASS:class_name})

def load_policy(module, class_name="Policy"):
  return load_object({MODULE:module, CLASS:class_name})

def load_algorithm(module, class_name="Algorithm"):
  return load_object({MODULE:module, CLASS:class_name})

def load_object(attrs):
  return load_class(attrs[MODULE], attrs[CLASS])

def load_class(module, class_name):
  return getattr(importlib.import_module(module), class_name)