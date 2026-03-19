import argparse

def add_arguments(parser: argparse.ArgumentParser):
    from .tracking import add_arguments as add_tracking_arguments
    from .taiyi import add_arguments as add_taiyi_arguments

    add_tracking_arguments(parser)
    add_taiyi_arguments(parser)
