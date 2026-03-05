import argparse
from torchvision import models

def get_args():
    parser = argparse.ArgumentParser(description="vit")

    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',
                        help='model architecture')

    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")

    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")

    parser.add_argument("--batch_size", default=256, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial (base) learning rate for train', dest='lr')
    
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay', dest='weight_decay')

    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")

    parser.add_argument("--workers", default=4, type=int,
                        help="number of data loading workers")

    parser.add_argument('--num_classes', default=1000, type=int,   
                    help='number of classes')

    return parser.parse_args()