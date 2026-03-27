"""
Root conftest.py — adds src/ to sys.path so all test files can import
modules_2photon, modules_histo, and modules_maldi directly.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
