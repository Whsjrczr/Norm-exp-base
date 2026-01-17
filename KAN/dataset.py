import torch



def equation(x, func):
    a = x[:, 0]
    b = x[:, 1]
    c = x[:, 2]

    if func == 1:
        return torch.cos(2 * torch.pi / (1 - a**2 - b**2 - c**2))
    elif func == 2:
        return 1 / (a + b + c)
    elif func == 3:
        return (a**2 + b**2 + c**2)
    elif func == 4:
        return 1 / (1 + a**2 + b**2 + c**2)
    elif func == 'f':
        return (a+b+c)/(1+a*b+ a*c + b*c)
    elif func == '3r':
        return (a**3 + b**3 + c**3)
    else:
        return (
            torch.sin(a * 2) + 0.5 * torch.cos(b * 3) -
            1.2 * b**2 + 3 * a * b -
            2.5 * torch.exp(-c) + 6 +
            b*c -
            3.5
        )
    
    

def prepare_data(seed, error=0.05):
    torch.manual_seed(seed=seed)  # 固定随机种子
    n_samples = 10000  # 样本数量
    x_data = torch.randn(n_samples, 3)  # 输入数据 (1000, 3)
    y_error = torch.randn(n_samples)*(error*2)-error
    y_true = equation(x_data)
    y_data = (y_true+y_error).unsqueeze(1)  # 输出数据，增加一个维度 (1000, 1)

    train_size = int(0.7 * n_samples)  # 70% 训练集
    val_size = int(0.15 * n_samples)  # 15% 验证集
    test_size = n_samples - train_size - val_size  # 剩下的 15% 测试集

    x_train, x_val, x_test = x_data[:train_size], x_data[train_size:train_size + val_size], x_data[train_size + val_size:]
    y_train, y_val, y_test = y_data[:train_size], y_data[train_size:train_size + val_size], y_data[train_size + val_size:]
    y_true_test = y_true[train_size + val_size:]

    return x_train, x_val, x_test, y_train, y_val, y_test, y_true_test.unsqueeze(1)