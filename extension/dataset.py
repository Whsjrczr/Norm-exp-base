import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
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
    group.add_argument('--dataset-root', metavar='PATH', default=os.path.expanduser('E:\\norm-exp\\dataset'), type=utils.path,
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


def _use_grey():
    if _config.dataset_cfg.get('grey') is not None:
        return bool(_config.dataset_cfg.get('grey'))
    if _config.dataset_cfg.get('nogrey') is not None:
        return not bool(_config.dataset_cfg.get('nogrey'))
    return False


def getDatasetConfigFlag():
    flag = ''
    flag += _config.dataset
    if str.find(_config.dataset, 'cifar10')>-1 or str.find(_config.dataset, 'ImageNet')>-1:
        if _use_grey():
            flag += '_grey'
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


def _is_vit_style_folder_dataset(args: argparse.Namespace):
    return _config.dataset_cfg.get('loader') == 'vit' or (
        hasattr(args, 'im_size') and len(getattr(args, 'im_size', [])) > 0 and args.dataset in ['ImageNet', 'folder']
    )


def _resolve_folder_image_size(args: argparse.Namespace):
    if hasattr(args, 'im_size') and len(args.im_size) > 0:
        if isinstance(args.im_size, (tuple, list)):
            return int(args.im_size[-1])
        return int(args.im_size)
    return int(_config.dataset_cfg.get('image_size', 224))


def _get_folder_normalize(args: argparse.Namespace):
    mean = _config.dataset_cfg.get('mean', [0.485, 0.456, 0.406])
    std = _config.dataset_cfg.get('std', [0.229, 0.224, 0.225])
    if hasattr(args, 'in_chans') and args.in_chans == 1 and len(mean) == 3 and len(std) == 3:
        mean = [mean[0]]
        std = [std[0]]
    return transforms.Normalize(mean=mean, std=std)


def _get_folder_transforms(args: argparse.Namespace, train: bool):
    if not _is_vit_style_folder_dataset(args):
        grey = _use_grey()
        if grey:
            transform_list = [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        return transforms.Compose(transform_list)

    image_size = _resolve_folder_image_size(args)
    val_resize_size = getattr(args, 'val_resize_size', _config.dataset_cfg.get('val_resize_size', int(image_size * 256 / 224)))
    normalize = _get_folder_normalize(args)

    if train:
        transform_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform_list = [
            transforms.Resize(val_resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]

    if getattr(args, 'in_chans', None) == 1:
        transform_list.insert(-2, transforms.Grayscale(num_output_channels=1))
    return transforms.Compose(transform_list)


def _build_folder_dataset(args: argparse.Namespace, root: str, train: bool, target_transform=None):
    split = 'train' if train else 'val'
    dataset_root = os.path.join(root, split)
    dataset = torchvision.datasets.ImageFolder(
        dataset_root,
        _get_folder_transforms(args, train),
        target_transform,
    )

    if _is_vit_style_folder_dataset(args):
        args.im_size = [_resolve_folder_image_size(args)]
    elif len(args.im_size) == 0:
        grey = _use_grey()
        args.im_size = (1, 256, 256) if grey else (3, 256, 256)

    if getattr(args, 'dataset_classes', None) in [None, 0]:
        args.dataset_classes = len(dataset.classes)
    return dataset, dataset_root


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
    
    grey = _use_grey()
    random_label = _config.dataset_cfg.get('random_label', False)
    
    if args.dataset != 'folder':
        root = os.path.join(root, args.dataset)  
    if args.dataset in ['mnist', 'fashion-mnist']:
        transform_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif args.dataset == 'cifar10':
        if grey:
            transform_list = [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ]
        transform_pipeline = transforms.Compose(transform_list)
    
    args.im_size = []
    #### TODO: add resize transform

    if args.dataset == 'mnist':
        if len(args.im_size) == 0:
            args.im_size = (1, 28, 28)
        args.dataset_classes = 10
        dataset = torchvision.datasets.mnist.MNIST(root, train, transform_pipeline, target_transform, download=True)
    elif args.dataset == 'fashion-mnist':
        if len(args.im_size) == 0:
            args.im_size = (1, 28, 28)
        args.dataset_classes = 10
        dataset = torchvision.datasets.FashionMNIST(root, train, transform_pipeline, target_transform, download=True)
    elif args.dataset == 'cifar10':
        if len(args.im_size) == 0:
            if grey:
                args.im_size = (1, 32, 32)
            else:
                args.im_size = (3, 32, 32)
        args.dataset_classes = 10
        dataset = torchvision.datasets.CIFAR10(root, train, transform_pipeline, target_transform, download=True)
    elif args.dataset in ['ImageNet', 'folder']:
        dataset, root = _build_folder_dataset(args, root, train, target_transform)
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
    batch_size = args.batch_size[not train]
    if not train and hasattr(args, 'val_batch_size') and args.val_batch_size:
        batch_size = args.val_batch_size
    shuffle = train and not getattr(args, 'disable_train_shuffle', False)
    drop_last = train
    if args.dataset in ['ImageNet', 'folder'] and _is_vit_style_folder_dataset(args):
        drop_last = bool(_config.dataset_cfg.get('drop_last_train', True)) if train else False
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs,
    )
    print("Successfully load dataset!")
    LOG = get_logger()
    LOG('==> Dataset: {}'.format(dataset))
    return dataset_loader
