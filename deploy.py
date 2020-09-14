import argparse, json, os

from services.deploy import Deployer

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--host")
PARSER.add_argument("--project_name")

if __name__ == "__main__":
  args = PARSER.parse_args()

  cwd = os.getcwd()

  deployer = Deployer(root_directory=cwd)

  deployer.clean()
  archive_path = deployer.package(folders=["services"], files=["config.json","main.py","requirements.txt"])
  deployer.deploy(host=args.host, project_name=args.project_name, archive_path=archive_path)