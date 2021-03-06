import argparse, json, os
from ethical_rl.deploy import Deployer

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--host")
PARSER.add_argument("--project_name", default="ethical_rl")

if __name__ == "__main__":
  args = PARSER.parse_args()

  cwd = os.getcwd()

  deployer = Deployer(root_directory=cwd)

  deployer.clean()
  archive_path = deployer.package(folders=["ethical_rl"], files=["config.json","main.py","default_config.json"])
  deployer.deploy(host=args.host, project_name=args.project_name, archive_path=archive_path)