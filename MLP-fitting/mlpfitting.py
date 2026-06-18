#!/usr/bin/env python3
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from fitting import FittingTrainer


if __name__ == "__main__":
    runner = FittingTrainer()
    runner.train()
