import importlib

def load_object(attrs):
  return load_class(attrs["module"], attrs["class"])

def load_class(module, class_name):
  return getattr(importlib.import_module(module), class_name)