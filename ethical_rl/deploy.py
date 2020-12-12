import shutil, os, time, subprocess
from ethical_rl.constants import *

class Deployer:
  def __init__(self, **kwargs):
    self.root_directory = kwargs.get(ROOT_DIRECTORY)

  def clean(self, **kwargs):
    shutil.rmtree(os.path.join(self.root_directory, DEPLOY),ignore_errors=True)
    os.mkdir(DEPLOY)

  def package(self, **kwargs):
    archive_name = f"{time.time()}".replace(".","")
    archive_path = os.path.join(self.root_directory, DEPLOY, archive_name)

    os.makedirs(archive_path, exist_ok=True)

    for f in kwargs.get(FILES) or []:
      shutil.copyfile(os.path.join(self.root_directory, f), os.path.join(archive_path, f))

    for f in kwargs.get(FOLDERS) or []:
      shutil.copytree(os.path.join(self.root_directory, f), os.path.join(archive_path, f))

    shutil.make_archive(archive_path, ZIP, archive_path)
    return archive_path

  def deploy(self, **kwargs):
    scp_args = [SCP, f"{kwargs.get(ARCHIVE_PATH)}.zip", f"{kwargs.get(HOST)}:~/uploads/{kwargs.get(PROJECT_NAME)}/"]
    subprocess.run(scp_args)

    ssh_args = [SSH, kwargs.get(HOST), f'unzip -o ~/uploads/{kwargs.get(PROJECT_NAME)}/{str(kwargs.get(ARCHIVE_PATH)).split("/")[-1]}', "-d", f"~/{kwargs.get(PROJECT_NAME)}"]
    subprocess.run(ssh_args)