import subprocess
from config import BASE_DIR

def install_bat_file():
    print("Preparing to run the batch file as administrator...")
    batch_file = r'D:\projects\matrix_finetune\resources\RunAsAdmin.bat'
    subprocess.run(['cmd.exe', '/c', batch_file], shell=True)
    print("Batch file execution complete.")

install_bat_file()
