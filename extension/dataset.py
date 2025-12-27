import argparse
import os
import torch
import torchvision
import torch.utils.data
from . import utils
from .logger import get_logger
from .utils import str2dict
from torchvision.datasets.folder import has_file_allowed_extension, default_loader, IMG_EXTENSIONS
import random
import numpy as np

# dataset_list = ['mnist', 'fashion-mnist', 'mnist_RandomLabel', 'fashion-mnist_RandomLabel', 'cifar10', 'cifar10_nogrey','cifar10_RandomLabel', 'ImageNet', 'folder']
dataset_list= ['mnist', 'fashion-mnist', 'cifar10', 'ImageNet', 'folder']


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Dataset Option')
    group.add_argument('--dataset', metavar='NAME', default='cifar10', choices=dataset_list,
                       help='The name of dataset in {' + ', '.join(dataset_list) + '}')
    group.add_argument('--dataset-cfg',type=str2dict, default={}, metavar='DICT', help='dataset config.')
    group.add_argument('--dataset-root', metavar='PATH', default=os.path.expanduser('/home/PycharmProjects/data/cifar10/'), type=utils.path,
                       help='The directory which contains needed dataset.')
    group.add_argument('-b', '--batch-size', type=utils.str2list, default=[64], metavar='NUMs',
                       help='The size of mini-batch')
    group.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='The number of data loading workers.')
    group.add_argument('--im-size', type=utils.str2tuple, default=(32, 32), metavar='NUMs',
                       help='Resize image to special size. (default: no resize)')
    group.add_argument('--dataset-classes', type=int, default=None, help='The number of classes in dataset.')
    return group

class _config:
    dataset = 'mnist'
    dataset_cfg = {}
    _methods = ['mnist', 'fashion-mnist', 'cifar10', 'ImageNet', 'folder']


def getDatasetConfigFlag():
    flag = ''
    flag += _config.dataset
    if str.find(_config.dataset, 'cifar10')>-1 or str.find(_config.dataset, 'ImageNet')>-1:
        if _config.dataset_cfg.get('nogrey') != None:
            flag += '_nogrey'
    if _config.dataset_cfg.get('random_label') != None:
        flag += '_RL'
    return flag


def setting(cfg: argparse.Namespace, **kwargs):
    for key, value in vars(cfg).items():
        if key in _config.__dict__:
            setattr(_config, key, value)
    flagname = getDatasetConfigFlag()
    return flagname


def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class DatasetFlatFolder(torch.utils.data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/xxx.ext
        root/xxy.ext
        root/xxz.ext

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable): A function to load a sample given its path.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, transform=None, loader=default_loader):
        samples = make_dataset(root, IMG_EXTENSIONS)
        assert len(samples) > 0, "Found 0 files in: " + root + "\nSupported extensions are: " + ",".join(IMG_EXTENSIONS)
        self.root = root
        self.loader = loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: 'sample' where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_dataset_loader(args: argparse.Namespace, target_transform=None, train=True, use_cuda=True):
    args.dataset_root = os.path.expanduser(args.dataset_root)
    root = args.dataset_root
    assert os.path.exists(root), 'Please assign the correct dataset root path with --dataset-root <PATH>'
    print("Successfully find the dataset root path")
    
    nogrey = _config.dataset_cfg.get('nogrey', False)
    random_label = _config.dataset_cfg.get('random_label', False)
    
    if args.dataset != 'folder':
        root = os.path.join(root, args.dataset)  
    if args.dataset in ['mnist', 'fashion-mnist']:
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
    elif args.dataset in ['cifar10', 'ImageNet', 'folder']:
        transform_list = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        if not nogrey:
            transform_list.append(torchvision.transforms.Grayscale())
        transforms = torchvision.transforms.Compose(transform_list)
    
    args.im_size = []
    #### TODO: add resize transform

    if args.dataset == 'mnist':
        if len(args.im_size) == 0:
            args.im_size = (1, 28, 28)
        args.dataset_classes = 10
        dataset = torchvision.datasets.mnist.MNIST(root, train, transforms, target_transform, download=True)
    elif args.dataset == 'fashion-mnist':
        if len(args.im_size) == 0:
            args.im_size = (1, 28, 28)
        args.dataset_classes = 10
        dataset = torchvision.datasets.FashionMNIST(root, train, transforms, target_transform, download=True)
    elif args.dataset == 'cifar10':
        if len(args.im_size) == 0:
            if nogrey:
                args.im_size = (3, 32, 32)
            else:
                args.im_size = (1, 32, 32)
        args.dataset_classes = 10
        dataset = torchvision.datasets.CIFAR10(root, train, transforms, target_transform, download=True)
    elif args.dataset in ['ImageNet', 'folder']:
        if len(args.im_size) == 0:
            if nogrey:
                args.im_size = (3, 256, 256)
            else:
                args.im_size = (1, 256, 256)
        args.dataset_classes = 1000
        root = os.path.join(root, 'train' if train else 'val')
        dataset = torchvision.datasets.ImageFolder(root, transforms, target_transform)
    else:
        raise FileNotFoundError('No such dataset')


    if random_label and hasattr(dataset, 'targets'):
        label_filename = f"{'train' if train else 'val'}_{args.dataset}_random_labels.npy"
        label_path = os.path.join(root, label_filename)
        if os.path.exists(label_path):
            random_labels = torch.from_numpy(np.load(label_path))
        else:
            random_labels = torch.from_numpy(np.random.randint(0, args.dataset_classes, size=len(dataset)))
            np.save(label_path, random_labels.numpy())
        dataset.targets = random_labels

    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    if len(args.batch_size) == 0:
        args.batch_size = [256, 256]
    elif len(args.batch_size) == 1:
        args.batch_size.append(args.batch_size[0])
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size[not train], shuffle=train,
                                                 drop_last=train, **loader_kwargs)
    print("Successfully load dataset!")
    LOG = get_logger()
    LOG('==> Dataset: {}'.format(dataset))
    return dataset_loader

