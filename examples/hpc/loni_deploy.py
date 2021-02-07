import argparse, json, os, sys
from ethical_rl.deploy import Deployer
from ethical_rl.constants import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--host", default="loni")
PARSER.add_argument("--project_name", default="ethical_rl")

if __name__ == "__main__":
  args = PARSER.parse_args()

  cwd = os.path.dirname(os.path.realpath(__file__))
  root_dir = os.getcwd()

  deployer = Deployer(root_directory=cwd)

  deployer.clean()

  # TODO: put in config, use regular deploy file
  files_to_deploy = [
    os.path.join(cwd, "create_job.sh"),
    os.path.join(root_dir, "config.json"),
    os.path.join(root_dir, "main.py"),
    os.path.join(root_dir, "default_config.json"),
    os.path.join(cwd, "requirements.txt")
  ]
  archive_path = deployer.package(files=files_to_deploy)
  deployer.deploy(host=args.host, project_name=args.project_name, archive_path=archive_path)

  chmod_args = [SSH, args.host, f'chmod 777 ~/{args.project_name}/create_job.sh']
  deployer.issue_cmd(chmod_args)