import os
import shutil
import subprocess


def run_shell_command(command: str, cwd: str = None) -> None:
    try:
        print(f"Running command:{command}")

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newline=True,
        )
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        stderr = process.stderr.read()
        if stderr:
            print(f"ERROR : {stderr}")
    except Exception as e:
        print(f"An error occured: {e}")


def create_folder(path: str, rm: bool = False) -> None:
    if rm:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
