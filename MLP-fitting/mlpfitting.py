from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import swanlab
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import random
import copy
import math
import os
import time
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

# from tomlkit import key


sys.path.append("..")

import shutil
import argparse
import extension as ext
from extension.normalization.pln import ParallelLN
from extension.stat_tracker import StatTracker


class MLP_Fitting:
    def __init__(self, in_dim=8, num_samples=512, cpu=False, target_dir=None):
        self.cfg = self.add_arguments()
        # print(ext.activation.setting(self.cfg))
        self.model_name = (
            "MLP_Fitting"
            + "_w"
            + str(self.cfg.arch_cfg["width"])
            + "_"
            + ext.activation.setting(self.cfg)
            + "_lr"
            + str(self.cfg.lr)
            + "_"
            + self.cfg.optimizer
            + "_epoch"
            + str(self.cfg.epochs)
            + "_seed"
            + str(self.cfg.seed)
        )
        # print(self.cfg.norm_cfg)
        self.width = self.cfg.arch_cfg["width"]
        self.lr = self.cfg.lr
        self.bs = self.cfg.batch_size[0]
        self.seed = self.cfg.seed
        self.in_dim = in_dim
        self.num_samples = num_samples
        self.target_dir = f"./results/{self.width}/{ext.activation.setting(self.cfg)}/{self.seed}_{self.lr}_{self.cfg.optimizer}"
        self.epochs = self.cfg.epochs
        self.get_dataset()
        self.set_seed(seed=self.seed)
        self.model = OneLayerNN(
            width=self.width,
            myNorm=nn.LayerNorm(self.width),
            myLayer=ext.Activation(self.width),
            in_dim=self.in_dim,
        )
        print(self.model)
        self.bestModel = OneLayerNN(
            width=self.width,
            myNorm=nn.LayerNorm(self.width),
            myLayer=ext.Activation(self.width),
            in_dim=self.in_dim,
        )
        self.optimizer = self.set_opt()
        self.criterion = nn.MSELoss()
        if cpu == False:
            self.device = torch.device(
                f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
        self.up_logger = swanlab
        self.logger_init()

    def set_opt(self):
        if self.cfg.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.cfg.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        return optimizer

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_dataset(self):
        self.set_seed(seed=0)
        self.X = 2 * torch.rand(self.num_samples, self.in_dim) - 1
        self.Y = 2 * torch.rand(self.num_samples, 1) - 1
        X = self.X.reshape(len(self.X), -1)
        Y = self.Y.reshape(len(self.Y), -1)
        dataset = TensorDataset(self.X, self.Y)
        train_dataset, _ = torch.utils.data.random_split(dataset, [self.num_samples, 0])
        self.train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=False)

    def train(self, warmup_epochs=5, lr_cosine_eta_min=1e-8):
        criterion = nn.MSELoss()
        self.model.to(self.device)
        losses = []
        self.best_loss = float("inf")
        self.scheduler_warmup = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs),
        )
        self.scheduler_cosine = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - warmup_epochs, eta_min=lr_cosine_eta_min
        )
        # 假设你用 PyTorch
        theta0 = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }
        self.para_start = torch.cat(
            [p.detach().clone().flatten() for p in self.model.parameters()]
        )
        theta_pre = None
        theta_ppre = None

        distances = []  # list of (epoch, distance)
        iterations = 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in self.train_loader:
                iterations += 1
                sum_sq = 0.0
                for name, param in self.model.named_parameters():
                    diff = param.detach().cpu() - theta0[name]
                    sum_sq += diff.view(-1).double().pow(2).sum().item()
                dist = sum_sq**0.5
                distances.append((epoch, dist))
                self.up_logger.log({"distance": dist}, step=epoch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if theta_ppre == None:
                    theta_ppre = theta_pre
                    theta_pre = {
                        name: param.detach().cpu().clone()
                        for name, param in self.model.named_parameters()
                    }
                else:
                    theta_now = {
                        name: param.detach().cpu().clone()
                        for name, param in self.model.named_parameters()
                    }
                    k1, k2, k3 = 0, 0, 0
                    for name, param in self.model.named_parameters():
                        k01 = param - theta_pre[name]
                        k02 = theta_pre[name] - theta_ppre[name]
                        k03 = k01 + k02
                        k1 += k01.view(-1).double().pow(2).sum().item()
                        k2 += k02.view(-1).double().pow(2).sum().item()
                        k3 += k03.view(-1).double().pow(2).sum().item()
                    curve_rate = (k1**0.5 + k2**0.5) / (k3**0.5 + 1e-14)
                    cosine_rate = 1 - (k1 * k2) / ((k1**0.5) + (k2**0.5) + 1e-14)
                    update = k2**0.5
                    self.up_logger.log(
                        {
                            "Curve Rate": curve_rate,
                            "Cosine Rate": cosine_rate,
                            "Update Rate": update,
                        },
                        step=epoch,
                    )
                    theta_ppre = theta_pre
                    theta_pre = theta_now
                total_loss += loss.item() * targets.size(0)
            avg_loss = total_loss / self.num_samples
            losses.append(avg_loss)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.bestModel = copy.deepcopy(self.model)
            print(
                f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Best Loss: {self.best_loss:.4f}, Learning Rate: {self.optimizer.param_groups[0]['lr']}"
            )
            self.up_logger.log(
                {
                    "Current Loss": avg_loss,
                    "Best Loss": self.best_loss,
                    "Learning Rate": self.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
            if epoch < warmup_epochs:
                self.scheduler_warmup.step()
            else:
                self.scheduler_cosine.step()
        print("best_loss:", self.best_loss)
        self.para_end = torch.cat(
            [p.detach().clone().flatten() for p in self.model.parameters()]
        )

    def logger_init(self):
        self.up_logger.login(
            api_key="xXNnMVoqSFYIA1nsQZhhd",
        )

        self.up_logger.init(
            project=self.cfg.exp_name,
            name="MLP_"
            + ext.activation.setting(self.cfg)
            + "_bs"
            + str(self.bs)
            + "_lr"
            + str(self.lr)
            + "_seed"
            + str(self.seed)
            + self.cfg.optimizer,
            config={
                "model": "mlp",
                "width": width,
                "activation": ext.activation.setting(self.cfg),
                "learning_rate": self.lr,
                "batch_size": self.bs,
                "seed": self.seed,
                "optimizer": self.cfg.optimizer,
                "epochs": self.epochs,
            },
        )

    def register_hook(self):
        self.hook_list = []
        depth = 0
        for name, module in self.model.named_modules():
            if (
                isinstance(module, DefaultAttention)
                or isinstance(module, LinearAttention)
                or isinstance(module, NonscaleLinearAttention)
                or isinstance(module, ItnAttention)
                or isinstance(module, GroupAttention)
                or isinstance(module, GWAttention)
            ):
                depth += 1
                module_name = "Attention Layer {}".format(depth)
                tracker = StatTracker(module, track_dic=self.track_dict)
                tracker.module_name = module_name
                tracker.register_hooks()
                self.hook_list.append(tracker)

    def add_arguments(self):
        parser = argparse.ArgumentParser("MLP Fitting")
        parser.add_argument(
            "-a",
            "--arch",
            metavar="ARCH",
            default="simple",
        )
        parser.add_argument(
            "--arch-cfg",
            metavar="DICT",
            default={},
            type=ext.utils.str2dict,
            help="The extra model architecture configuration.",
        )
        parser.add_argument(
            "-A",
            "--augmentation",
            type=ext.utils.str2bool,
            default=True,
            metavar="BOOL",
            help="Use data augmentation? (default: True)",
        )
        parser.add_argument(
            "-debug",
            "--debug",
            type=int,
            default=0,
            metavar="INT",
            help="For debug, as a step size",
        )
        parser.add_argument("--exp_name", type=str, default="0")
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=200)
        ext.dataset.add_arguments(parser)
        parser.set_defaults(dataset="cifar10", workers=4)
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method="steps", lr_steps=[100, 150], lr=0.1)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer="sgd", weight_decay=1e-4)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.visualization.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.attention.add_arguments(parser)
        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(
                namespace=ext.checkpoint.Checkpoint.load_config(args.resume)
            )
        return args


class OneLayerNN(nn.Module):
    def __init__(self, width, myNorm, myLayer, in_dim):
        super(OneLayerNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), myNorm, myLayer, nn.Linear(width, 1)
        )

    def forward(self, input: torch.FloatTensor):
        return self.net(input)


class Tee:
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self._terminal = sys.stdout  # 备份原始stdout

    def write(self, data):
        # 同时写入文件和终端
        self.fileobj.write(data)
        self._terminal.write(data)

    def flush(self):
        # 确保都能立即写出
        self.fileobj.flush()
        self._terminal.flush()


class stor_best:
    def __init__(self):
        self.dict = {}

    def update(self, af, lr, bs, opt, seed, loss):
        key = af
        setting = {"bs": bs, "lr": lr, "opt": opt, "seed": seed, "best_loss": loss}

        self.dict[key] = setting


def random_isosceles_right_directions_exact(theta0, thetav, seed=None):
    """
    保证：u + v = thetav - theta0
          ||u|| = ||v|| = ||h|| / sqrt(2)
          u · v = 0
    """
    if seed is not None:
        torch.manual_seed(seed)

    h = thetav - theta0
    L = torch.norm(h)
    if L == 0:
        raise ValueError("theta0 和 thetav 不能相同！")

    m = h / 2.0
    ell = L / 2.0

    # 随机生成垂直于 h 的单位向量 n
    n = torch.randn_like(h)
    n = n - torch.dot(n, h) * h / (L * L)
    n = n / torch.norm(n)

    u = m + ell * n
    v = m - ell * n

    return u, v


def compute_loss(theta, model, loss_fn, x_batch, y_batch):
    """把参数向量 theta 塞回模型，计算标量 loss（不破坏原模型）"""
    # 1. 备份原始参数
    orig_params = [p.clone() for p in model.parameters()]
    
    # 2. 写回新参数
    param_shapes = [p.shape for p in model.parameters()]
    split = torch.split(theta, [np.prod(s) for s in param_shapes])
    shaped = [s.reshape(shp) for s, shp in zip(split, param_shapes)]
    for p, val in zip(model.parameters(), shaped):
        p.data.copy_(val)
    
    # 3. 计算 loss
    out = model(x_batch)
    loss = loss_fn(out, y_batch).item()
    
    # 4. 恢复原始参数
    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)
    
    return loss


def loss_landscape_slice(
    theta0,
    u,
    v,
    model,
    loss_fn,
    x_batch,
    y_batch,
    grid_size=40,
    s_range=[-1/3, 4/3],
    t_range=[-1/3, 4/3],
    save_path=None,  # 新增：保存路径
    show_thetav=True,  # 新增：是否标注 θᵥ
):
    """
    在 theta0 + s*u + t*v 平面上采样 grid_size x grid_size 的 loss
    返回 (S, T, Loss) 三个 ndarray
    并可选：保存图像 + 标注 θ₀ 和 θᵥ
    """
    s_lin = torch.linspace(s_range[0], s_range[1], grid_size)
    t_lin = torch.linspace(t_range[0], t_range[1], grid_size)
    S, T = torch.meshgrid(s_lin, t_lin, indexing="ij")
    Loss = torch.zeros_like(S)
    theta0_flat = theta0

    for i in range(grid_size):
        for j in range(grid_size):
            theta = theta0_flat + S[i, j] * u + T[i, j] * v
            Loss[i, j] = compute_loss(theta, model, loss_fn, x_batch, y_batch)

    S_np, T_np, Loss_np = S.numpy(), T.numpy(), Loss.numpy()

    # ==================== 可视化 & 保存 ====================
    if save_path or show_thetav:
        plt.figure(figsize=(8, 7))
        Loss_np_safe = np.clip(Loss_np, 1e-14, None)  # 防止 log(0)

        # 使用 LogNorm 绘制
        cs = plt.contourf(
            S_np, T_np, Loss_np_safe,
            levels=60,
            cmap="viridis",
            norm=LogNorm(vmin=Loss_np_safe.min(), vmax=Loss_np_safe.max()),
            alpha=0.9
        )

        # 等高线也用 log 刻度
        plt.contour(
            S_np, T_np, Loss_np_safe,
            levels=np.logspace(np.log10(Loss_np_safe.min()), np.log10(Loss_np_safe.max()), 15),
            colors="white", alpha=0.3, linewidths=0.5
        )

        # colorbar 自动显示 log 刻度
        cbar = plt.colorbar(cs, shrink=0.8, format='%.2e')  # 科学计数法
        cbar.set_label("Loss (log scale)", fontsize=12)

        # 标注 θ₀
        plt.scatter(
            0, 0, c="red", s=120, marker="o", edgecolors="white", linewidth=2, zorder=5
        )
        plt.text(
            0,
            0,
            "  θ₀",
            color="white",
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=6,
        )

        # 标注 θᵥ（如果 u + v = thetav - theta0）
        if show_thetav:
            plt.scatter(
                1,
                1,
                c="lime",
                s=150,
                marker='*',
                edgecolors="black",
                linewidth=1,
                zorder=5,
            )
            plt.text(
                1,
                1,
                "  θᵥ",
                color="white",
                fontsize=14,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=6,
            )
            # 画斜边
            plt.plot([0, 1], [0, 1], "w--", alpha=0.7, linewidth=2, label="u + v")

        plt.xlabel("s (along u)", fontsize=12)
        plt.ylabel("t (along v)", fontsize=12)
        plt.title("Loss Landscape Slice – Isosceles Right Triangle", fontsize=14)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 自动生成保存路径
        if save_path is None:
            os.makedirs("landscape_plots", exist_ok=True)
            save_path = f"landscape_plots/loss_landscape.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"图像已保存：{save_path}")

    def verify_loss(theta, name):
        loss = compute_loss(theta, fm.model, loss_fn, x_batch, y_batch)
        print(f"{name} loss = {loss:.6f}")
        return loss

    print("验证两个端点：")
    l0 = verify_loss(theta0, "θ₀ (para_start)")
    lv = verify_loss(thetav, "θᵥ (para_end)")

    assert lv < l0, f"错误：θᵥ loss ({lv}) 应小于 θ₀ loss ({l0})！"
    print("通过！θᵥ 确实更优")

    return S_np, T_np, Loss_np

def plot_3d_landscape(S_np, T_np, Loss_np, save_path="landscape_3d.pdf", log_z=True):
    """
    绘制 3D loss landscape
    log_z=True → z轴对数刻度
    """
    # 防止 log(0)
    Loss_safe = np.clip(Loss_np, 1e-8, None)

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 曲面图
    if log_z:
        norm = LogNorm(vmin=Loss_safe.min(), vmax=Loss_safe.max())
        surf = ax.plot_surface(
            S_np, T_np, Loss_safe,
            cmap='viridis',
            norm=norm,
            linewidth=0, antialiased=False, alpha=0.9
        )
    else:
        surf = ax.plot_surface(
            S_np, T_np, Loss_np,
            cmap='viridis',
            linewidth=0, antialiased=False, alpha=0.9
        )

    # 标注 θ₀ 和 θᵥ
    i0 = np.argmin(np.abs(S_np[0]))      # s=0 的列索引
    j0 = np.argmin(np.abs(T_np[:, 0]))   # t=0 的行索引
    ax.scatter(0, 0, Loss_np[j0, i0], color='red', s=100, label='θ₀', depthshade=True)

    i1 = np.argmin(np.abs(S_np[0] - 1))   # s=1 的列索引
    j1 = np.argmin(np.abs(T_np[:, 0] - 1)) # t=1 的行索引
    ax.scatter(1, 1, Loss_np[j1, i1], color='lime', s=100, label='θᵥ', depthshade=True)

    # 坐标轴标签
    ax.set_xlabel('s (along u)', fontsize=12)
    ax.set_ylabel('t (along v)', fontsize=12)
    if log_z:
        ax.set_zlabel('Loss (log scale)', fontsize=12)
    else:
        ax.set_zlabel('Loss', fontsize=12)

    # z 轴对数刻度
    if log_z:
        ax.set_zscale('log')
        z_min, z_max = Loss_safe.min(), Loss_safe.max()
        ax.set_zlim(z_min, z_max)
        # 设置 log 刻度
        from matplotlib.ticker import LogLocator, LogFormatter
        ax.zaxis.set_major_locator(LogLocator(base=10.0))
        ax.zaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))

    # 标题
    ax.set_title('3D Loss Landscape – Isosceles Right Triangle', fontsize=14)

    # 颜色条
    cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Loss', fontsize=12)

    # 图例
    ax.legend()

    # 视角
    ax.view_init(elev=30, azim=225)  # 可调视角

    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D 图像已保存：{save_path}")


if __name__ == "__main__":
    ext.MagnitudeDebug.reset()
    torch.set_num_threads(4)
    in_dim = 8
    num_samples = 512

    width = 512
    cpu = True
    device_id = 3

    dic = stor_best()
    fm = MLP_Fitting(in_dim=in_dim, num_samples=num_samples, cpu=cpu)
    target_dir = fm.target_dir
    os.makedirs(target_dir, exist_ok=True)
    f = open(f"{target_dir}/log.txt", "w")
    backup = sys.stdout
    sys.stdout = Tee(f)
    # print("in_dim:", in_dim)
    # print("num_samples:", num_samples)
    # print("width:", width)
    # print("bs:", bs)
    # print("lr:", lr)
    # print("seed:", seed)
    # print("opt:", opt)
    # print("af:", af)

    fm.train()
    sys.stdout = backup
    f.close()
    checkpoint = {
        "model_dict": fm.bestModel.state_dict(),
        "in_dim": in_dim,
        "width": width,
        "bs": fm.bs,
        "lr": fm.lr,
        "seed": fm.seed,
        "opt": fm.cfg.optimizer,
        "af": ext.activation.setting(fm.cfg),
        "loss": fm.best_loss,
    }

    torch.save(checkpoint, f"{target_dir}/checkpoint.pth")
    best_loss = float("inf")

    if fm.best_loss < best_loss:
        dic.update(
            af=ext.activation.setting(fm.cfg),
            lr=fm.lr,
            bs=fm.bs,
            opt=fm.cfg.optimizer,
            seed=fm.seed,
            loss=fm.best_loss,
        )
        best_loss = fm.best_loss
        target_dir_best = f"./results/{fm.width}/{ext.activation.setting(fm.cfg)}/best"
        os.makedirs(target_dir_best, exist_ok=True)
        import shutil

        shutil.copy2(
            f"{target_dir}/log.txt",
            f"{target_dir_best}/log.txt",
        )
        shutil.copy2(
            f"{target_dir}/checkpoint.pth",
            f"{target_dir_best}/checkpoint.pth",
        )
    fm.up_logger.finish()

    f = open(f"./result/{width}/best.log", "w")
    backup = sys.stdout
    sys.stdout = Tee(f)
    print(dic.dict)
    sys.stdout = backup
    f.close()

    # draw landscape
    theta0 = fm.para_start
    thetav = fm.para_end
    loss_fn = nn.MSELoss()
    u, v = random_isosceles_right_directions_exact(theta0, thetav, seed=42)
    batch_iter = iter(fm.train_loader)
    x_batch, y_batch = next(batch_iter)

    # ---------- 3. 计算切面 ----------
    S, T, Loss = loss_landscape_slice(
        theta0, u, v,
        fm.model, loss_fn,
        x_batch, y_batch,
        grid_size=100,
        # s_range=[-5,6],
        # t_range=[-5,6],
        s_range=[-1/3,4/3],
        t_range=[-1/3,4/3],
        save_path="output.pdf"   # 指定路径，或留空自动保存
    )
    time.sleep(1)
    plt.clf()

    # 你的 S, T, Loss 已经是 numpy
    # plot_3d_landscape(S, T, Loss, save_path="output_3d.pdf", log_z=True)
