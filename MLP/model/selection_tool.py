from model import *
import torchvision.transforms as transforms

def get_model(model_name, model_width, model_depth, dataset, dropout_prob=0):
    if dataset in ('mnist', 'fashion-mnist', 'mnist_RandomLabel', 'fashion-mnist_RandomLabel'):
        input_size = 28 * 28
        output_size = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif dataset in ('cifar10', 'cifar10_RandomLabel'):
        input_size = 32 * 32
        output_size = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),transforms.Grayscale()])
    elif dataset == 'cifar10_nogrey':
        input_size = 32 * 32 * 3
        output_size = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if model_name == 'MLP':
        model_out = MLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob = dropout_prob)
    elif model_name == 'CenDropScalingMLP':
        model_out = CenDropScalingMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == 'CenDropScalingPreNormMLP':
        model_out = CenDropScalingPreNormMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == 'PreNormMLP':
        model_out = PreNormMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size,  dropout_prob = dropout_prob)
    elif model_name =='resnet18':
        model_out = resnet18(num_classes=output_size)
    elif model_name =='resnet34':
        model_out = resnet34(num_classes=output_size)
    elif model_name =='resnet50':
        model_out = resnet50(num_classes=output_size)
    elif model_name == 'ResCenDropScalingMLP':
        model_out = ResCenDropScalingMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    elif model_name == 'ResMLP':
        model_out = ResMLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    else:
        model_out = MLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    return model_out, transform