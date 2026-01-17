import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
from model.KAN_layer import *
from model.MLP import *
import wandb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import random
import numpy as np
from dataset import prepare_data, equation
from model_selection import get_model



# 定义训练函数
def train_model(model, layers_hidden, norm, upgrade_grid, num_epochs=20, lr=0.01, momentum=0.9, seed=1, weight_decay=0, error=0.05):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x_train, x_val, x_test, y_train, y_val, y_test, y_true_test = prepare_data(seed=seed, error=error)


    print("Dataset Done!")

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr,weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/100)
# 

    csv_filename = f"{model._get_name()}_{norm.__name__}_e{num_epochs}_lr{lr}_{upgrade_grid}_wd{weight_decay}_"
    hidden = ''
    for size in layers_hidden:
        hidden = hidden + str(size) + "_"
    csv_filename = csv_filename + hidden + f"seeds{seed}_{type(optimizer).__name__}_error{error}"
    print(csv_filename)
    wandb.init(
        project="KAN_wd",
        name=csv_filename,
        config={
            "model": model._get_name(),
            "hidden_layer": hidden,
            "epoch": num_epochs,
            "upgrade_grid": update_grid,
            "norm": norm.__name__,
            "lr": lr,
            "seed": seed,
            "optimizer": type(optimizer).__name__,
            "weight_decay": weight_decay,
            "error": error,
            # "eltma": (lr/100),
        }
    )
 
    csv_filename = "./equal_new/" + csv_filename + ".png"



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    print(model)


    # 初始化记录损失和准确率的列表
    train_losses, test_losses = [], []
    train_accuracies, accuracies = [], []

    print("Begin")

    # 训练模型
    for epoch in range(num_epochs):
        model.train()

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        optimizer.zero_grad()
        output = model(x_train)
        if torch.isnan(output).any():
            print("train_exit")
            exit(0)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        train_r2 = compute_r2(output.detach(), y_train.clone())

        train_losses.append(loss.item())
        train_accuracies.append(train_r2)

        model.eval()
        with torch.no_grad():
            
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_true_test = y_true_test.to(device)
            optimizer.zero_grad()
            output = model(x_val)
            if torch.isnan(output).any():
                print("val_exit")
                exit(0)
            loss = criterion(output, y_val)
            loss_true = criterion(output, y_true_test.clone())
            r2 = compute_r2(output, y_val.clone())
            r2_true = compute_r2(output, y_true_test.clone())
            
        test_losses.append(loss.item())
        accuracies.append(r2)

        wandb.log({"epoch": epoch, "train_loss": train_losses[-1], "train_r2": train_accuracies[-1], "test_loss": test_losses[-1], "test_r2": accuracies[-1], "true_loss": loss_true, "true_r2": r2_true, "lr": optimizer.param_groups[0]['lr']})

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, Train R2: {train_accuracies[-1]:.2f}, "
              f"Test Loss: {test_losses[-1]:.4f}, Test R2: {accuracies[-1]:.2f}, "
              f"True Test Loss: {loss_true:.4f}, True Test R2: {r2_true:.2f}")
        

    # 将结果保存到CSV文件
    
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])
    #     for epoch in range(num_epochs):
    #         writer.writerow([epoch + 1, train_losses[epoch], train_accuracies[epoch], test_losses[epoch], accuracies[epoch]])

    # print(f"Results saved to {csv_filename}")

    model.eval()
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        output = model(x_test)
        r2 = compute_r2(output, y_test)

    output = output.cpu()
    y_test = y_test.cpu()

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    # 设置空心点：facecolor='none' 表示填充颜色为空，edgecolor 设置边缘颜色
    plt.scatter(y_test.numpy(), output.numpy(), alpha=0.6, label=f'KAN Predictions (R2={r2:.4f})',
                facecolor='none', edgecolor='blue', linewidth=1)  # 空心蓝色点
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.legend()

    x_test = x_test.to('cpu')
    model.to('cpu')

    x_fixed = x_test.clone()  # 克隆测试数据，避免修改原数据
    x_fixed[:, 1] = 0.0  # 固定 x2 为 0
    x_fixed[:, 2] = 0.0  # 固定 x3 为 0

    # 为 x[:, 0] 生成连续值
    x_fixed[:, 0] = torch.linspace(x_test[:, 0].min(), x_test[:, 0].max(), x_test.size(0))

    # 计算真实值和预测值
    y_true_curve = equation(x_fixed).numpy()  # 真实值
    y_kan_curve = model(x_fixed)
    loss = criterion(y_kan_curve, y_test)

    y_kan_curve = y_kan_curve.detach().numpy()  # KAN 模型预测值
    r2 = r2_score(y_kan_curve, y_test)


    # 可视化
    plt.subplot(1, 2, 2)

    plt.plot(x_fixed[:, 0].numpy(), y_true_curve, label="True Curve", color="blue", linestyle="--", linewidth=2)
    plt.plot(x_fixed[:, 0].numpy(), y_kan_curve, label="KAN Predictions", color="red", linestyle="-", linewidth=2)
    plt.xlabel("x[:, 0] (One Input Feature)")
    plt.ylabel("Output (y)")
    plt.title("Visualization of True Curve and Predictions")
    plt.legend()
    plt.grid(True)



    plt.tight_layout()
    plt.savefig(csv_filename)

    plt.close()

    wandb.finish()

def compute_r2(y_true, y_pred):
    return r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

# 主函数
if __name__ == "__main__":

    layers_hidden = [3, 16, 32, 16, 1]
    norm = nn.Identity
    update_grid = False
    weight_decay = 0
    lr = 0.01
    error_list = [0, 0.05, 0.01]
    for error in error_list:
        model = get_model("MLP", layers_hidden, norm, update_grid)
        train_model(model, layers_hidden, norm, update_grid, num_epochs=1000, weight_decay=weight_decay, lr=lr, error=error)
  

    layers_hidden = [3, 16, 32, 16, 1]
    model_names_list = ["KAN"]
    norm_list_list = [[nn.Identity, nn.BatchNorm1d, nn.LayerNorm]]
    update_grid= False
    weight_decay_list = [0, 0.1, 0.05, 0.01,0.001]
    lr_list = [0.01]
    seed = 1
    error_list = [0, 0.05, 0.01]
    # dropout_rate_list = [0, 0.1, 0.05, 0.01]



    for model_name, norm_list in zip(model_names_list, norm_list_list):
        for norm in norm_list:
            for lr in lr_list:
                for error in error_list:
                    for weight_decay in weight_decay_list:
                        model = get_model(model_name, layers_hidden, norm, update_grid)
                        train_model(model, layers_hidden, norm, update_grid, num_epochs=1000, weight_decay=weight_decay, lr=lr, error=error)




