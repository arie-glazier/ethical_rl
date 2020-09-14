import shutil, os, time, subprocess

class Deployer:
  def __init__(self, **kwargs):
    self.root_directory = kwargs.get("root_directory")

  def clean(self, **kwargs):
    shutil.rmtree(os.path.join(self.root_directory, "deploy"),ignore_errors=True)
    os.mkdir("deploy")

  def package(self, **kwargs):
    archive_name = f"{time.time()}".replace(".","")
    archive_path = os.path.join(self.root_directory, "deploy", archive_name)

    os.makedirs(archive_path, exist_ok=True)

    for f in kwargs.get("files") or []:
      shutil.copyfile(os.path.join(self.root_directory, f), os.path.join(archive_path, f))

    for f in kwargs.get("folders") or []:
      shutil.copytree(os.path.join(self.root_directory, f), os.path.join(archive_path, f))

    shutil.make_archive(archive_path, "zip", archive_path)
    return archive_path

  def deploy(self, **kwargs):
    scp_args = ["scp", f"{kwargs.get('archive_path')}.zip", f"{kwargs.get('host')}:~/uploads/{kwargs.get('project_name')}/"]
    subprocess.run(scp_args)

    ssh_args = ["ssh", kwargs.get("host"), f'unzip -o ~/uploads/{kwargs.get("project_name")}/{str(kwargs.get("archive_path")).split("/")[-1]}', "-d", f"~/{kwargs.get('project_name')}"]
    subprocess.run(ssh_args)