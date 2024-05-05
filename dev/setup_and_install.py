import ctypes
import os
import subprocess
import sys
from config import BASE_DIR

# Unfortunately, this version doesn't work but the .sh file works.

def install_nvidia_cuda():
    print("Preparing to run the batch file as administrator...")
    batch_file = os.path.join(BASE_DIR, 'resources', 'RunAsAdmin.bat')
    subprocess.run(['cmd.exe', '/c', batch_file], shell=True)
    print("Batch file execution complete.")


def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute: {command} | Error: {str(e)}")
        return False
    return True


def install_package(package):
    return run_command(f"{sys.executable} -m pip install {package}")


def install_torch():
    torch_command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    print("Installing PyTorch, TorchVision, and TorchAudio...")
    return run_command(torch_command)


def conda_install_cuda_dependencies():
    print("Running batch file to set up CUDA dependencies...")
    result = subprocess.run(["cmd.exe", "/c", os.path.join(BASE_DIR, 'resources', 'setup_env.bat')], capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to install dependencies:")
        print(result.stdout)
        print(result.stderr)
        return False
    else:
        print("Dependencies installed successfully.")
        return True


def install_transformers():
    return install_package("transformers")


def install_additional_packages():
    return install_package("trl datasets")


def install_triton():
    triton_dir = "./triton"
    if not os.path.exists(triton_dir):
        print("Cloning Triton repository...")
        clone_command = f"git clone https://github.com/openai/triton.git"
        if not run_command(clone_command):
            return False

    print("Installing build dependencies...")
    deps_command = f"{sys.executable} -m pip install ninja cmake wheel"
    if not run_command(deps_command):
        return False

    print("Installing Triton from source...")
    install_command = f"{sys.executable} -m pip install -e {os.path.join(triton_dir, 'python')}"
    return run_command(install_command)


def install_orchestrator(install_cuda=True):
    if install_cuda:
        install_nvidia_cuda()
    else:
        print("Skipping CUDA installation because install_cuda is set to False.")

    install_torch()

    import torch
    if torch.cuda.is_available():
        steps = [
            ('CUDA Dependencies', conda_install_cuda_dependencies),
            ('Hugging Face Transformers', install_transformers),
            ('Additional Packages', install_additional_packages),
        ]
    else:
        print("Skipping CUDA-specific installations because CUDA is not available.")
        steps = [
            ('Hugging Face Transformers', install_transformers),
            ('Additional Packages', install_additional_packages),
        ]

    for step_name, step_func in steps:
        print(f"Starting installation step: {step_name}")
        if not step_func():
            print(f"Installation step '{step_name}' failed. Please resolve the issues and restart the installation from this step.")
            break
        print(f"Completed installation step: {step_name}")

    ensure_imports_work()
    return True

def ensure_imports_work():
    try:
        import torch
        import trl
        import transformers
        import datasets
    except ImportError as e:
        print(f"Failed to import a required package. Error: {str(e)}")
        return False

    print(f"The following imports worked: torch, trl, transformers, datasets")
    return True




if __name__ == "__main__":
    install_orchestrator(install_cuda=False)
    print("Installation complete.")
    #install_triton()


"""
# For the HTTP/REST client
pip install tritonclient[http]

# For the GRPC client
pip install tritonclient[grpc]

"""
