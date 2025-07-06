import os
import sys

def add_sopath(PYBIND11_DIR=None):
    if PYBIND11_DIR == None:
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        PYBIND11_DIR = os.path.join(parent_dir, "build","lib")
    
    if PYBIND11_DIR not in sys.path:
        sys.path.append(PYBIND11_DIR)
