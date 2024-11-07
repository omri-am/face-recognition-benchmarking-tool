import subprocess
import sys

def install_pytorch():
    command = [
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', '-q'
    ]
    
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing PyTorch: {e}")

install_pytorch()

from .baseModel import *
from .multiModelTaskManager import *
from .baseTask import *
from .plotHelper import *
