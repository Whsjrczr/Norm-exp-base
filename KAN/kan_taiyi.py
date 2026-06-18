#!/usr/bin/env python3
import sys

import torch

from KAN import KANTrainer


def _ensure_flag(argv, flag):
    if flag in argv:
        return argv
    return argv + [flag]


if __name__ == "__main__":
    sys.argv = _ensure_flag(sys.argv, "--taiyi")
    trainer = KANTrainer()
    torch.set_num_threads(1)
    trainer.train()
