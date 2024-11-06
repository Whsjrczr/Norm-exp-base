import argparse

def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Taiyi Options')
    group.add_argument('--taiyi', action='store_true', help='Is the visualization training process?')
    group.add_argument('--vis-env', default=None, help='The env name of visdom use. Default: <model_name>')
    return
