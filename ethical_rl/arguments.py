import os, json, argparse
from ethical_rl.constants import *

# This class is so we can override anything in
# config.json from the command line
class Arguments:
  def __init__(self, **kwargs):
    self.parser = argparse.ArgumentParser()
    self.pwd = kwargs[PWD]

    self.available_args = json.loads(open(os.path.join(self.pwd, "default_config.json")).read()).keys()
    for arg in self.available_args:
      self.parser.add_argument(f"--{arg}")