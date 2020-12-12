import os, json
from ethical_rl.constants import *

# unify defaults, tests, args (reverse precedence)
class Config:
  def __init__(self, **kwargs):
    self.pwd = kwargs[PWD]
    self.args_dict = {k:v for k,v in vars(kwargs[ARGS]).items() if v }
    self.full_config = json.loads(open(os.path.join(self.pwd, "config.json")).read())
    self.default_config = json.loads(open(os.path.join(self.pwd,"default_config.json")).read())
    self.test_config = self.full_config.get(self.args_dict.get(TEST_NAME), {})
    self.config = {**self.default_config, **self.test_config, **self.args_dict}