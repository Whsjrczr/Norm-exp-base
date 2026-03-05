import argparse

def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Wandb Options')
    parser.add_argument('--visualize', action='store_true', help='use wandb for visualization and logging')
    parser.add_argument('--wandb-project', type=str, default='test', help='wandb project name')
    group.add_argument('--taiyi', action='store_true', help='Is the training process under the monitor of taiyi?')
    return
