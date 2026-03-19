import argparse

from .tracking import VisdomVisualizer as Visualization
from .tracking import normalize_config, setting_visdom as setting


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Visdom Options")
    group.add_argument("--visdom", action="store_true", help="enable Visdom visualization")
    group.add_argument("--vis", action="store_true", help="legacy alias of --visdom")
    group.add_argument("--visdom-port", "--vis-port", dest="vis_port", default=6006, type=int, help="Visdom port")
    group.add_argument("--visdom-env", "--vis-env", dest="vis_env", default=None, help="Visdom environment name")
