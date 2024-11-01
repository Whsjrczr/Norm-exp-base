from model import *
import torchvision.transforms as transforms

def get_model(model_name, model_width, model_depth, dataset):
    if dataset == 'mnist' or 'fashion-mnist' or 'mnist_RandomLabel' or 'fashion-mnist_RandomLabel':
        input_size = 28 * 28
        output_size = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif dataset == 'cifar10' or 'cifar10_RandomLabel':
        input_size = 32 * 32
        output_size = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),transforms.Grayscale()])

    if model_name == 'MLP':
        model_out = MLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name == 'MLPReLU':
        model_out = MLPReLU(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name =='LinearModel':
        model_out = LinearModel(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name =='Linear':
        model_out = Linear(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    elif model_name =='resnet18':
        model_out = resnet18(num_classes=output_size)
    elif model_name =='resnet34':
        model_out = resnet34(num_classes=output_size)
    elif model_name =='resnet50':
        model_out = resnet50(num_classes=output_size)
    else:
        model_out = MLP(width=model_width, depth=model_depth, input_size=input_size, output_size=output_size)
    return model_out, transform