import shutil, os, time, subprocess
from ethical_rl.constants import *

class Deployer:
  def __init__(self, **kwargs):
    self.root_directory = kwargs.get(ROOT_DIRECTORY)

  def clean(self, **kwargs):
    deploy_folder = os.path.join(self.root_directory, DEPLOY)
    if os.path.exists(deploy_folder): shutil.rmtree(deploy_folder, ignore_errors=True)
    os.mkdir(deploy_folder)

  def package(self, **kwargs):
    archive_name = f"{time.time()}".replace(".","")
    archive_path = os.path.join(self.root_directory, DEPLOY, archive_name)

    os.makedirs(archive_path, exist_ok=True)

    # expecting absolute paths here
    for f in kwargs.get(FILES) or []:
      shutil.copyfile(f, os.path.join(archive_path, os.path.basename(f)))

    for f in kwargs.get(FOLDERS) or []:
      shutil.copytree(f, os.path.join(archive_path, os.path.basename(f)))

    shutil.make_archive(archive_path, ZIP, archive_path)
    return archive_path

  def deploy(self, **kwargs):
    scp_args = [SCP, f"{kwargs.get(ARCHIVE_PATH)}.zip", f"{kwargs.get(HOST)}:~/uploads/{kwargs.get(PROJECT_NAME)}/"]
    subprocess.run(scp_args)

    ssh_args = [SSH, kwargs.get(HOST), f'unzip -o ~/uploads/{kwargs.get(PROJECT_NAME)}/{str(kwargs.get(ARCHIVE_PATH)).split("/")[-1]}', "-d", f"~/{kwargs.get(PROJECT_NAME)}"]
    subprocess.run(ssh_args)

  def issue_cmd(self, cmd_args):
    return subprocess.run(cmd_args)