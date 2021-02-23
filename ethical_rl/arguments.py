import os, json, argparse
from ethical_rl.constants import *

# This class is so we can override anything in
# config.json from the command line
class Arguments:
  def __init__(self, **kwargs):
    self.parser = argparse.ArgumentParser()
    self.pwd = kwargs[PWD]

    self.default_config = json.loads(open(os.path.join(self.pwd, "default_config.json")).read())
    self.available_args = self.default_config.keys()
    for arg in self.available_args:
      arg_def = f"--{arg}"
      self.parser.add_argument(arg_def)