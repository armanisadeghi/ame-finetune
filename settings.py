# Project Root /settings.py
# This module should be imported into most scripts to set up the environment and global settings.

import os
import warnings
import logging

# Define the base directory and other global settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAINING_SOURCE_DIR = os.path.join(DATA_DIR, 'training')

# Set the compiler to be used for C extensions
os.environ['CC'] = '/usr/bin/gcc'

warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
