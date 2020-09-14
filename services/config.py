import os, json

# unify defaults, tests, args (reverse precedence)
class Config:
  def __init__(self, **kwargs):
    self.pwd = kwargs["pwd"]
    self.args_dict = {k:v for k,v in vars(kwargs["args"]).items() if v }
    self.full_config = json.loads(open(os.path.join(self.pwd, "config.json")).read())
    self.default_config = self.full_config["DEFAULT_CONFIG"]
    self.test_config = self.full_config.get(self.args_dict.get("test_name"), {})
    self.config = {**self.default_config, **self.test_config, **self.args_dict}