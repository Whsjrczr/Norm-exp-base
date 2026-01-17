import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from kan import KANLayer as KANLinear


class KANLinearReLU(torch.nn.Module):
    def __init__(
            self,
            in_feature: int,
            out_feature: int,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.01,
            scale_spline: float = 0.3,           
            scale_base: float = 0.3, 
            enable_spline_scale: bool = False,
            base_activation = nn.SiLU,
            grid_range: list = [-1,1],
            grid_eps: float = 1e-4, #0.02
            device: str = 'cpu',
            eps: float=1e-5,
    ):
        """
        初始化KAN层。

        参数:
            in_feature (int): 层输入特征数。
            out_feature (int): 层输出特征数。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数，常用k表示。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            # grid_eps (float): 网格更新时的小偏移量。
            # grid_range (list): 网格范围。
        """
        super(KANLinearReLU, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.eps =eps

        """
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        # 生成线性间隔的网格点，增加一维，在维度上扩展为in_dim行
        grid = extend_grid(grid, k_extend=k)
        # 在网格的两端扩展 spline_order 个点
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        ## 样条函数的基函数是不用更新的，依据网格生成。
        """
        self.grid_gap = (grid_range[1] - grid_range[0]) / grid_size
        
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * self.grid_gap
                + grid_range[0]
            )
            .expand(in_feature, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # residual connection
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_feature, in_feature))

        # spline weight
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_feature, in_feature, grid_size + spline_order)
        )

        if enable_spline_scale:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_feature, in_feature)
            )
        self.enable_spline_scale = enable_spline_scale

        # 噪声缩放系数，用于初始化样条权重时添加噪声
        self.scale_noise = scale_noise
        # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        self.scale_base = scale_base
        # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        self.scale_spline = scale_spline

        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()
        # 网格更新时的小偏移量，用于在更新网格时引入微小变化，避免过拟合
        self.grid_eps = grid_eps
        # 存疑
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        # TODO：考虑换成别的初始化
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_feature, self.out_feature)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_spline_scale else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            
            if self.enable_spline_scale:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # 作者此前使用了一般的初始化，效果不佳
                # 使用kaiming_uniform_方法初始化样条缩放参数spline_scaler
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    @property
    def scaled_spline_weight(self):
        """
        计算带有缩放因子的样条权重。

        样条缩放：如果启用了 enable_standalone_scale_spline，
        则将 spline_scaler 张量扩展一维后与 spline_weight 相乘，
        否则直接返回 spline_weight。

        具体来说，spline_weight 是一个三维张量，形状为 (out_features, in_features, grid_size + spline_order)。
        而 spline_scaler 是一个二维张量，形状为 (out_features, in_features)。
        为了使 spline_scaler 能够与 spline_weight 逐元素相乘，
        需要将 spline_scaler 的最后一维扩展，以匹配 spline_weight 的第三维。

        返回:
            torch.Tensor: 带有缩放因子的样条权重张量。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_spline_scale
            else 1.0
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        assert x.size(-1) == self.in_feature
        
        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 将输入张量展平为二维
        x = x.view(-1, self.in_feature)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # 计算B样条基函数的输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_feature, -1),
        )
        
        # 合并基础输出和样条输出
        output = base_output + spline_output
        # output = spline_output
        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_feature)
        
        return output
    
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        update_grid 方法用于根据输入数据动态更新B样条的网格点，从而适应输入数据的分布。
        该方法通过重新计算和调整网格点，确保B样条基函数能够更好地拟合数据。
        这在训练过程中可能会提高模型的精度和稳定性。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 网格更新的边缘大小，用于在更新网格时引入微小变化。
        """
        # 确保输入张量的维度正确
        assert x.dim() == 2 and x.size(1) == self.in_feature
        batch = x.size(0)  # 获取批量大小

        # 计算输入张量的B样条基函数
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # 转置为 (in, batch, coeff)

        # 获取当前的样条权重
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置为 (in, coeff, out)

        # 计算未缩减的样条输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 为了收集数据分布，对每个通道分别进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀步长，并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 混合均匀网格和自适应网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包括样条边界
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新模型中的网格点
        self.grid.copy_(grid.T)

        # 重新计算样条权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是对论文中提到的原始L1正则化的一种简单模拟，因为原始方法需要从
        展开的 (batch, in_features, out_features) 中间张量计算绝对值和熵，
        但如果我们想要一个高效的内存实现，这些张量会被隐藏在F.linear函数后面。

        现在的L1正则化计算为样条权重的平均绝对值。
        作者的实现还包括这个项，此外还有基于样本的正则化。
        """
        # 计算样条权重的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)
        
        # 计算激活正则化损失，即所有样条权重绝对值的和
        regularization_loss_activation = l1_fake.sum()
        
        # 计算每个权重占总和的比例
        p = l1_fake / regularization_loss_activation
        
        # 计算熵正则化损失，即上述比例的负熵
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        # 返回总的正则化损失，包含激活正则化和熵正则化
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature

        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # 为了进行逐元素操作，将输入张量的最后一维扩展一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        base = x - grid
        base = self.relu(base)
        base = (base[:, :, 0:-4]-2*base[:, :, 2:-2]+base[:, :, 4:])**3 - 4 * (base[:, :, 1:-3] - 2 * base[:, :, 2:-2] + base[:, :, 3:-1])**3
        base = base / (6 * self.grid_gap ** 3)
        assert base.size() == (
            x.size(0),
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        return base.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线的系数。
        curve2coeff 方法用于计算插值给定点的曲线的系数。
        这些系数用于表示插值曲线在特定点的形状和位置。
        具体来说，该方法通过求解线性方程组来找到B样条基函数在给定点上的插值系数。
        此方法的作用是根据输入和输出点计算B样条基函数的系数，
        使得这些基函数能够精确插值给定的输入输出点对。
        这样可以用于拟合数据或在模型中应用非线性变换。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature
        
        # 确保输出张量的形状正确
        assert y.size() == (x.size(0), self.in_feature, self.out_feature)

        # 计算B样条基函数
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        
        # 转置输出张量
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # 使用线性代数方法求解线性方程组，找到系数
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        
        # 调整结果的形状
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        # 确保结果张量的形状正确
        assert result.size() == (
            self.out_feature,
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        
        # 返回连续存储的结果张量
        return result.contiguous()


class KANLinearNoRes(torch.nn.Module):
    def __init__(
            self,
            in_feature: int,
            out_feature: int,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.01,
            scale_spline: float = 0.3,           
            scale_base: float = 0.3, 
            enable_spline_scale: bool = False,
            base_activation = nn.SiLU,
            grid_range: list = [-1,1],
            grid_eps: float = 1e-4, #0.02
            device: str = 'cpu',
            eps: float=1e-5,
    ):
        """
        初始化KAN层。

        参数:
            in_feature (int): 层输入特征数。
            out_feature (int): 层输出特征数。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数，常用k表示。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            # grid_eps (float): 网格更新时的小偏移量。
            # grid_range (list): 网格范围。
        """
        super(KANLinearNoRes, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.eps =eps

        """
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        # 生成线性间隔的网格点，增加一维，在维度上扩展为in_dim行
        grid = extend_grid(grid, k_extend=k)
        # 在网格的两端扩展 spline_order 个点
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        ## 样条函数的基函数是不用更新的，依据网格生成。
        """
        grid_gap = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * grid_gap
                + grid_range[0]
            )
            .expand(in_feature, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # residual connection
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_feature, in_feature))

        # spline weight
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_feature, in_feature, grid_size + spline_order)
        )

        if enable_spline_scale:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_feature, in_feature)
            )
        self.enable_spline_scale = enable_spline_scale

        # 噪声缩放系数，用于初始化样条权重时添加噪声
        self.scale_noise = scale_noise
        # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        self.scale_base = scale_base
        # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        self.scale_spline = scale_spline

        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()
        # 网格更新时的小偏移量，用于在更新网格时引入微小变化，避免过拟合
        self.grid_eps = grid_eps
        # 存疑

        self.reset_parameters()

    def reset_parameters(self):
        # TODO：考虑换成别的初始化
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_feature, self.out_feature)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_spline_scale else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            
            if self.enable_spline_scale:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # 作者此前使用了一般的初始化，效果不佳
                # 使用kaiming_uniform_方法初始化样条缩放参数spline_scaler
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    @property
    def scaled_spline_weight(self):
        """
        计算带有缩放因子的样条权重。

        样条缩放：如果启用了 enable_standalone_scale_spline，
        则将 spline_scaler 张量扩展一维后与 spline_weight 相乘，
        否则直接返回 spline_weight。

        具体来说，spline_weight 是一个三维张量，形状为 (out_features, in_features, grid_size + spline_order)。
        而 spline_scaler 是一个二维张量，形状为 (out_features, in_features)。
        为了使 spline_scaler 能够与 spline_weight 逐元素相乘，
        需要将 spline_scaler 的最后一维扩展，以匹配 spline_weight 的第三维。

        返回:
            torch.Tensor: 带有缩放因子的样条权重张量。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_spline_scale
            else 1.0
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        assert x.size(-1) == self.in_feature
        
        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 将输入张量展平为二维
        x = x.view(-1, self.in_feature)

        # 计算基础线性变换的输出

        # 计算B样条基函数的输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_feature, -1),
        )
        
        # 合并基础输出和样条输出
        output = spline_output
        # output = spline_output
        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_feature)
        
        return output
    

    # @torch.no_grad()
    # def weight_norm(self):
    #     base_weight = self.base_weight - torch.mean(self.base_weight, dim=1, keepdim=True)
    #     base_weight = base_weight / torch.sqrt(self.eps + torch.var(base_weight, dim=1, keepdim=True))
    #     self.base_weight.copy_(base_weight)

    #     spline_weight = self.spline_weight - torch.mean(self.spline_weight, dim=1, keepdim=True)
    #     spline_weight = spline_weight / torch.sqrt(self.eps + torch.var(spline_weight, dim=1, keepdim=True))
    #     self.spline_weight.copy_(spline_weight)
    
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        update_grid 方法用于根据输入数据动态更新B样条的网格点，从而适应输入数据的分布。
        该方法通过重新计算和调整网格点，确保B样条基函数能够更好地拟合数据。
        这在训练过程中可能会提高模型的精度和稳定性。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 网格更新的边缘大小，用于在更新网格时引入微小变化。
        """
        # 确保输入张量的维度正确
        assert x.dim() == 2 and x.size(1) == self.in_feature
        batch = x.size(0)  # 获取批量大小

        # 计算输入张量的B样条基函数
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # 转置为 (in, batch, coeff)

        # 获取当前的样条权重
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置为 (in, coeff, out)

        # 计算未缩减的样条输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 为了收集数据分布，对每个通道分别进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀步长，并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 混合均匀网格和自适应网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包括样条边界
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新模型中的网格点
        self.grid.copy_(grid.T)

        # 重新计算样条权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是对论文中提到的原始L1正则化的一种简单模拟，因为原始方法需要从
        展开的 (batch, in_features, out_features) 中间张量计算绝对值和熵，
        但如果我们想要一个高效的内存实现，这些张量会被隐藏在F.linear函数后面。

        现在的L1正则化计算为样条权重的平均绝对值。
        作者的实现还包括这个项，此外还有基于样本的正则化。
        """
        # 计算样条权重的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)
        
        # 计算激活正则化损失，即所有样条权重绝对值的和
        regularization_loss_activation = l1_fake.sum()
        
        # 计算每个权重占总和的比例
        p = l1_fake / regularization_loss_activation
        
        # 计算熵正则化损失，即上述比例的负熵
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        # 返回总的正则化损失，包含激活正则化和熵正则化
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature

        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # 为了进行逐元素操作，将输入张量的最后一维扩展一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # 迭代计算样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        
        # 确保B样条基函数的输出形状正确
        assert bases.size() == (
            x.size(0),
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线的系数。
        curve2coeff 方法用于计算插值给定点的曲线的系数。
        这些系数用于表示插值曲线在特定点的形状和位置。
        具体来说，该方法通过求解线性方程组来找到B样条基函数在给定点上的插值系数。
        此方法的作用是根据输入和输出点计算B样条基函数的系数，
        使得这些基函数能够精确插值给定的输入输出点对。
        这样可以用于拟合数据或在模型中应用非线性变换。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature
        
        # 确保输出张量的形状正确
        assert y.size() == (x.size(0), self.in_feature, self.out_feature)

        # 计算B样条基函数
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        
        # 转置输出张量
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # 使用线性代数方法求解线性方程组，找到系数
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        
        # 调整结果的形状
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        # 确保结果张量的形状正确
        assert result.size() == (
            self.out_feature,
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        
        # 返回连续存储的结果张量
        return result.contiguous()


class KANLinearNorm(torch.nn.Module):
    def __init__(
            self,
            in_feature: int,
            out_feature: int,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.01,
            scale_spline: float = 0.3,           
            scale_base: float = 0.3, 
            enable_spline_scale: bool = False,
            base_activation = nn.SiLU,
            grid_range: list = [-1,1],
            grid_eps: float = 1e-4, #0.02
            device: str = 'cpu',
            eps: float=1e-5,
            init: str = 'origin',
            use_norm = nn.LayerNorm,
    ):
        """
        初始化KAN层。

        参数:
            in_feature (int): 层输入特征数。
            out_feature (int): 层输出特征数。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数，常用k表示。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            # grid_eps (float): 网格更新时的小偏移量。
            # grid_range (list): 网格范围。
        """
        super(KANLinearNorm, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.eps =eps
        self.norm = use_norm([self.in_feature, self.grid_size + self.spline_order])

        """
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        # 生成线性间隔的网格点，增加一维，在维度上扩展为in_dim行
        grid = extend_grid(grid, k_extend=k)
        # 在网格的两端扩展 spline_order 个点
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        ## 样条函数的基函数是不用更新的，依据网格生成。
        """
        grid_gap = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * grid_gap
                + grid_range[0]
            )
            .expand(in_feature, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # residual connection
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_feature, in_feature))

        # spline weight
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_feature, in_feature, grid_size + spline_order)
        )

        if enable_spline_scale:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_feature, in_feature)
            )
        self.enable_spline_scale = enable_spline_scale

        # 噪声缩放系数，用于初始化样条权重时添加噪声
        self.scale_noise = scale_noise
        # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        self.scale_base = scale_base
        # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        self.scale_spline = scale_spline

        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()
        # 网格更新时的小偏移量，用于在更新网格时引入微小变化，避免过拟合
        self.grid_eps = grid_eps
        # 存疑

        self.reset_parameters()
        if init == "xavier":
            torch.nn.init.xavier_uniform_(self.spline_weight,gain=nn.init.calculate_gain('relu'))
        elif init == "kaiming":
            torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def reset_parameters(self):
        # TODO：考虑换成别的初始化
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_feature, self.out_feature)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_spline_scale else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            
            if self.enable_spline_scale:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # 作者此前使用了一般的初始化，效果不佳
                # 使用kaiming_uniform_方法初始化样条缩放参数spline_scaler
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    @property
    def scaled_spline_weight(self):
        """
        计算带有缩放因子的样条权重。

        样条缩放：如果启用了 enable_standalone_scale_spline，
        则将 spline_scaler 张量扩展一维后与 spline_weight 相乘，
        否则直接返回 spline_weight。

        具体来说，spline_weight 是一个三维张量，形状为 (out_features, in_features, grid_size + spline_order)。
        而 spline_scaler 是一个二维张量，形状为 (out_features, in_features)。
        为了使 spline_scaler 能够与 spline_weight 逐元素相乘，
        需要将 spline_scaler 的最后一维扩展，以匹配 spline_weight 的第三维。

        返回:
            torch.Tensor: 带有缩放因子的样条权重张量。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_spline_scale
            else 1.0
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        # print(x.size(-1))
        assert x.size(-1) == self.in_feature
        
        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 将输入张量展平为二维
        x = x.view(-1, self.in_feature)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        x_b = self.b_splines(x)

        # 计算B样条基函数的输出
        spline_output = F.linear(
            math.sqrt(self.in_feature) * x_b.view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_feature, -1),
        )
        
        # 合并基础输出和样条输出
        output = base_output + spline_output
        # output = spline_output
        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_feature)
        
        return output
    

    # @torch.no_grad()
    # def weight_norm(self):
    #     base_weight = self.base_weight - torch.mean(self.base_weight, dim=1, keepdim=True)
    #     base_weight = base_weight / torch.sqrt(self.eps + torch.var(base_weight, dim=1, keepdim=True))
    #     self.base_weight.copy_(base_weight)

    #     spline_weight = self.spline_weight - torch.mean(self.spline_weight, dim=1, keepdim=True)
    #     spline_weight = spline_weight / torch.sqrt(self.eps + torch.var(spline_weight, dim=1, keepdim=True))
    #     self.spline_weight.copy_(spline_weight)
    
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        update_grid 方法用于根据输入数据动态更新B样条的网格点，从而适应输入数据的分布。
        该方法通过重新计算和调整网格点，确保B样条基函数能够更好地拟合数据。
        这在训练过程中可能会提高模型的精度和稳定性。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 网格更新的边缘大小，用于在更新网格时引入微小变化。
        """
        # 确保输入张量的维度正确
        assert x.dim() == 2 and x.size(1) == self.in_feature
        batch = x.size(0)  # 获取批量大小

        # 计算输入张量的B样条基函数
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # 转置为 (in, batch, coeff)

        # 获取当前的样条权重
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置为 (in, coeff, out)

        # 计算未缩减的样条输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 为了收集数据分布，对每个通道分别进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀步长，并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 混合均匀网格和自适应网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包括样条边界
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新模型中的网格点
        self.grid.copy_(grid.T)

        # 重新计算样条权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是对论文中提到的原始L1正则化的一种简单模拟，因为原始方法需要从
        展开的 (batch, in_features, out_features) 中间张量计算绝对值和熵，
        但如果我们想要一个高效的内存实现，这些张量会被隐藏在F.linear函数后面。

        现在的L1正则化计算为样条权重的平均绝对值。
        作者的实现还包括这个项，此外还有基于样本的正则化。
        """
        # 计算样条权重的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)
        
        # 计算激活正则化损失，即所有样条权重绝对值的和
        regularization_loss_activation = l1_fake.sum()
        
        # 计算每个权重占总和的比例
        p = l1_fake / regularization_loss_activation
        
        # 计算熵正则化损失，即上述比例的负熵
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        # 返回总的正则化损失，包含激活正则化和熵正则化
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature

        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # 为了进行逐元素操作，将输入张量的最后一维扩展一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # 迭代计算样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        
        # 确保B样条基函数的输出形状正确
        assert bases.size() == (
            x.size(0),
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线的系数。
        curve2coeff 方法用于计算插值给定点的曲线的系数。
        这些系数用于表示插值曲线在特定点的形状和位置。
        具体来说，该方法通过求解线性方程组来找到B样条基函数在给定点上的插值系数。
        此方法的作用是根据输入和输出点计算B样条基函数的系数，
        使得这些基函数能够精确插值给定的输入输出点对。
        这样可以用于拟合数据或在模型中应用非线性变换。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature
        
        # 确保输出张量的形状正确
        assert y.size() == (x.size(0), self.in_feature, self.out_feature)

        # 计算B样条基函数
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        
        # 转置输出张量
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # 使用线性代数方法求解线性方程组，找到系数
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        
        # 调整结果的形状
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        # 确保结果张量的形状正确
        assert result.size() == (
            self.out_feature,
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        
        # 返回连续存储的结果张量
        return result.contiguous()





class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
    ):
        """
        初始化KAN模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            grid_eps (float): 网格更新时的小偏移量。
            grid_range (list): 网格范围。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
                # KANLinear(
                #     in_features,
                #     out_features,
                #     grid_size,
                #     spline_order,
                #     noise_scale=scale_noise,
                #     scale_base_sigma=scale_base,
                #     scale_sp=scale_spline,
                #     base_fun=base_activation,
                #     grid_eps=grid_eps,
                #     grid_range=grid_range,
                # )
                
            )



    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if self.update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

class KAN_noRes(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
    ):
        """
        初始化KAN模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            grid_eps (float): 网格更新时的小偏移量。
            grid_range (list): 网格范围。
        """
        super(KAN_noRes, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinearNoRes(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )



    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if self.update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

class KANLinear_WN(torch.nn.Module):
    def __init__(
            self,
            in_feature: int,
            out_feature: int,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.01,
            scale_spline: float = 0.3,           
            scale_base: float = 0.3, 
            enable_spline_scale: bool = False,
            base_activation = nn.SiLU,
            grid_range: list = [-1,1],
            grid_eps: float = 1e-4, #0.02
            device: str = 'cpu',
            eps: float=1e-5,
    ):
        """
        初始化KAN层。

        参数:
            in_feature (int): 层输入特征数。
            out_feature (int): 层输出特征数。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数，常用k表示。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            # grid_eps (float): 网格更新时的小偏移量。
            # grid_range (list): 网格范围。
        """
        super(KANLinear_WN, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.eps =eps

        """
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        # 生成线性间隔的网格点，增加一维，在维度上扩展为in_dim行
        grid = extend_grid(grid, k_extend=k)
        # 在网格的两端扩展 spline_order 个点
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        ## 样条函数的基函数是不用更新的，依据网格生成。
        """
        grid_gap = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * grid_gap
                + grid_range[0]
            )
            .expand(in_feature, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # residual connection
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_feature, in_feature))

        # spline weight
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_feature, in_feature, grid_size + spline_order)
        )
        self.g_b = nn.Parameter(torch.ones(out_feature))  # 缩放因子
        self.g_s = nn.Parameter(torch.ones(out_feature))  # 缩放因子

        if enable_spline_scale:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_feature, in_feature)
            )
        self.enable_spline_scale = enable_spline_scale

        # 噪声缩放系数，用于初始化样条权重时添加噪声
        self.scale_noise = scale_noise
        # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        self.scale_base = scale_base
        # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        self.scale_spline = scale_spline

        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()
        # 网格更新时的小偏移量，用于在更新网格时引入微小变化，避免过拟合
        self.grid_eps = grid_eps
        # 存疑

        self.reset_parameters()

    def reset_parameters(self):
        # TODO：考虑换成别的初始化
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_feature, self.out_feature)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_spline_scale else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            
            if self.enable_spline_scale:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # 作者此前使用了一般的初始化，效果不佳
                # 使用kaiming_uniform_方法初始化样条缩放参数spline_scaler
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def scaled_spline_weight(self, x):
        """
        计算带有缩放因子的样条权重。

        样条缩放：如果启用了 enable_standalone_scale_spline，
        则将 spline_scaler 张量扩展一维后与 spline_weight 相乘，
        否则直接返回 spline_weight。

        具体来说，spline_weight 是一个三维张量，形状为 (out_features, in_features, grid_size + spline_order)。
        而 spline_scaler 是一个二维张量，形状为 (out_features, in_features)。
        为了使 spline_scaler 能够与 spline_weight 逐元素相乘，
        需要将 spline_scaler 的最后一维扩展，以匹配 spline_weight 的第三维。

        返回:
            torch.Tensor: 带有缩放因子的样条权重张量。
        """
        return x * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_spline_scale
            else 1.0
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        assert x.size(-1) == self.in_feature
        
        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 将输入张量展平为二维
        x = x.view(-1, self.in_feature)

        norm_b = torch.norm(self.base_weight, dim=1, keepdim=True)
        b_weight = self.g_b.view(-1, 1) * (self.base_weight / norm_b)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), b_weight)

        norm_s = torch.norm(self.spline_weight, dim=1, keepdim=True)
        s_weight = self.g_s.view(-1, 1, 1) * (self.spline_weight / norm_s)
        ss_weight = self.scaled_spline_weight(s_weight)
        
        # 计算B样条基函数的输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            ss_weight.view(self.out_feature, -1),
        )
        
        # 合并基础输出和样条输出
        output = base_output + spline_output
        
        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_feature)
        
        return output
    

    # @torch.no_grad()
    # def weight_norm(self):
    #     base_weight = self.base_weight - torch.mean(self.base_weight, dim=1, keepdim=True)
    #     base_weight = base_weight / torch.sqrt(self.eps + torch.var(base_weight, dim=1, keepdim=True))
    #     self.base_weight.copy_(base_weight)

    #     spline_weight = self.spline_weight - torch.mean(self.spline_weight, dim=1, keepdim=True)
    #     spline_weight = spline_weight / torch.sqrt(self.eps + torch.var(spline_weight, dim=1, keepdim=True))
    #     self.spline_weight.copy_(spline_weight)
    
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        update_grid 方法用于根据输入数据动态更新B样条的网格点，从而适应输入数据的分布。
        该方法通过重新计算和调整网格点，确保B样条基函数能够更好地拟合数据。
        这在训练过程中可能会提高模型的精度和稳定性。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 网格更新的边缘大小，用于在更新网格时引入微小变化。
        """
        # 确保输入张量的维度正确
        assert x.dim() == 2 and x.size(1) == self.in_feature
        batch = x.size(0)  # 获取批量大小

        # 计算输入张量的B样条基函数
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # 转置为 (in, batch, coeff)

        norm_s = torch.norm(self.spline_weight, dim=1, keepdim=True)
        s_weight = self.g_s.view(-1, 1, 1) * (self.spline_weight / norm_s)
        # 获取当前的样条权重
        orig_coeff = self.scaled_spline_weight(s_weight=s_weight)  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置为 (in, coeff, out)

        # 计算未缩减的样条输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 为了收集数据分布，对每个通道分别进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀步长，并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 混合均匀网格和自适应网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包括样条边界
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新模型中的网格点
        self.grid.copy_(grid.T)

        # 重新计算样条权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是对论文中提到的原始L1正则化的一种简单模拟，因为原始方法需要从
        展开的 (batch, in_features, out_features) 中间张量计算绝对值和熵，
        但如果我们想要一个高效的内存实现，这些张量会被隐藏在F.linear函数后面。

        现在的L1正则化计算为样条权重的平均绝对值。
        作者的实现还包括这个项，此外还有基于样本的正则化。
        """
        # 计算样条权重的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)
        
        # 计算激活正则化损失，即所有样条权重绝对值的和
        regularization_loss_activation = l1_fake.sum()
        
        # 计算每个权重占总和的比例
        p = l1_fake / regularization_loss_activation
        
        # 计算熵正则化损失，即上述比例的负熵
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        # 返回总的正则化损失，包含激活正则化和熵正则化
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature

        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # 为了进行逐元素操作，将输入张量的最后一维扩展一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # 迭代计算样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        # 确保B样条基函数的输出形状正确
        assert bases.size() == (
            x.size(0),
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线的系数。
        curve2coeff 方法用于计算插值给定点的曲线的系数。
        这些系数用于表示插值曲线在特定点的形状和位置。
        具体来说，该方法通过求解线性方程组来找到B样条基函数在给定点上的插值系数。
        此方法的作用是根据输入和输出点计算B样条基函数的系数，
        使得这些基函数能够精确插值给定的输入输出点对。
        这样可以用于拟合数据或在模型中应用非线性变换。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature
        
        # 确保输出张量的形状正确
        assert y.size() == (x.size(0), self.in_feature, self.out_feature)

        # 计算B样条基函数
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        
        # 转置输出张量
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # 使用线性代数方法求解线性方程组，找到系数
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        
        # 调整结果的形状
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        # 确保结果张量的形状正确
        assert result.size() == (
            self.out_feature,
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        
        # 返回连续存储的结果张量
        return result.contiguous()


class KANLinear_WN_new(torch.nn.Module):
    def __init__(
            self,
            in_feature: int,
            out_feature: int,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.01,
            scale_spline: float = 0.3,           
            scale_base: float = 0.3, 
            enable_spline_scale: bool = False,
            base_activation = nn.SiLU,
            grid_range: list = [-1,1],
            grid_eps: float = 1e-4, #0.02
            device: str = 'cpu',
            eps: float=1e-5,
    ):
        """
        初始化KAN层。

        参数:
            in_feature (int): 层输入特征数。
            out_feature (int): 层输出特征数。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数，常用k表示。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            # grid_eps (float): 网格更新时的小偏移量。
            # grid_range (list): 网格范围。
        """
        super(KANLinear_WN_new, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.eps =eps

        """
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        # 生成线性间隔的网格点，增加一维，在维度上扩展为in_dim行
        grid = extend_grid(grid, k_extend=k)
        # 在网格的两端扩展 spline_order 个点
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        ## 样条函数的基函数是不用更新的，依据网格生成。
        """
        grid_gap = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * grid_gap
                + grid_range[0]
            )
            .expand(in_feature, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # residual connection
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_feature, in_feature))

        # spline weight
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_feature, in_feature, grid_size + spline_order)
        )
        self.g_b = nn.Parameter(torch.ones(in_feature))  # 缩放因子
        self.g_s = nn.Parameter(torch.ones(in_feature))  # 缩放因子

        if enable_spline_scale:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_feature, in_feature)
            )
        self.enable_spline_scale = enable_spline_scale

        # 噪声缩放系数，用于初始化样条权重时添加噪声
        self.scale_noise = scale_noise
        # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        self.scale_base = scale_base
        # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        self.scale_spline = scale_spline

        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()
        # 网格更新时的小偏移量，用于在更新网格时引入微小变化，避免过拟合
        self.grid_eps = grid_eps
        # 存疑

        self.reset_parameters()

    def reset_parameters(self):
        # TODO：考虑换成别的初始化
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_feature, self.out_feature)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_spline_scale else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            
            if self.enable_spline_scale:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # 作者此前使用了一般的初始化，效果不佳
                # 使用kaiming_uniform_方法初始化样条缩放参数spline_scaler
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def scaled_spline_weight(self, x):
        """
        计算带有缩放因子的样条权重。

        样条缩放：如果启用了 enable_standalone_scale_spline，
        则将 spline_scaler 张量扩展一维后与 spline_weight 相乘，
        否则直接返回 spline_weight。

        具体来说，spline_weight 是一个三维张量，形状为 (out_features, in_features, grid_size + spline_order)。
        而 spline_scaler 是一个二维张量，形状为 (out_features, in_features)。
        为了使 spline_scaler 能够与 spline_weight 逐元素相乘，
        需要将 spline_scaler 的最后一维扩展，以匹配 spline_weight 的第三维。

        返回:
            torch.Tensor: 带有缩放因子的样条权重张量。
        """
        return x * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_spline_scale
            else 1.0
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        assert x.size(-1) == self.in_feature
        
        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 将输入张量展平为二维
        x = x.view(-1, self.in_feature)

        norm_b = torch.norm(self.base_weight, dim=0, keepdim=True)
        b_weight = self.g_b.view(1, -1) * (self.base_weight / norm_b)
        
        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), b_weight)

        norm_s = torch.norm(self.spline_weight, dim=0, keepdim=True)
        s_weight = self.g_s.view(1, -1, 1) * (self.spline_weight / norm_s)
        ss_weight = self.scaled_spline_weight(s_weight)
        
        # 计算B样条基函数的输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            ss_weight.view(self.out_feature, -1),
        )
        
        # 合并基础输出和样条输出
        output = base_output + spline_output
        
        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_feature)
        
        return output
    

    # @torch.no_grad()
    # def weight_norm(self):
    #     base_weight = self.base_weight - torch.mean(self.base_weight, dim=1, keepdim=True)
    #     base_weight = base_weight / torch.sqrt(self.eps + torch.var(base_weight, dim=1, keepdim=True))
    #     self.base_weight.copy_(base_weight)

    #     spline_weight = self.spline_weight - torch.mean(self.spline_weight, dim=1, keepdim=True)
    #     spline_weight = spline_weight / torch.sqrt(self.eps + torch.var(spline_weight, dim=1, keepdim=True))
    #     self.spline_weight.copy_(spline_weight)
    
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        update_grid 方法用于根据输入数据动态更新B样条的网格点，从而适应输入数据的分布。
        该方法通过重新计算和调整网格点，确保B样条基函数能够更好地拟合数据。
        这在训练过程中可能会提高模型的精度和稳定性。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 网格更新的边缘大小，用于在更新网格时引入微小变化。
        """
        # 确保输入张量的维度正确
        assert x.dim() == 2 and x.size(1) == self.in_feature
        batch = x.size(0)  # 获取批量大小

        # 计算输入张量的B样条基函数
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # 转置为 (in, batch, coeff)

        norm_s = torch.norm(self.spline_weight, dim=1, keepdim=True)
        s_weight = self.g_s.view(-1, 1, 1) * (self.spline_weight / norm_s)
        # 获取当前的样条权重
        orig_coeff = self.scaled_spline_weight(s_weight=s_weight)  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置为 (in, coeff, out)

        # 计算未缩减的样条输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 为了收集数据分布，对每个通道分别进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀步长，并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 混合均匀网格和自适应网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包括样条边界
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新模型中的网格点
        self.grid.copy_(grid.T)

        # 重新计算样条权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是对论文中提到的原始L1正则化的一种简单模拟，因为原始方法需要从
        展开的 (batch, in_features, out_features) 中间张量计算绝对值和熵，
        但如果我们想要一个高效的内存实现，这些张量会被隐藏在F.linear函数后面。

        现在的L1正则化计算为样条权重的平均绝对值。
        作者的实现还包括这个项，此外还有基于样本的正则化。
        """
        # 计算样条权重的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)
        
        # 计算激活正则化损失，即所有样条权重绝对值的和
        regularization_loss_activation = l1_fake.sum()
        
        # 计算每个权重占总和的比例
        p = l1_fake / regularization_loss_activation
        
        # 计算熵正则化损失，即上述比例的负熵
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        # 返回总的正则化损失，包含激活正则化和熵正则化
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature

        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # 为了进行逐元素操作，将输入张量的最后一维扩展一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # 迭代计算样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        # 确保B样条基函数的输出形状正确
        assert bases.size() == (
            x.size(0),
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线的系数。
        curve2coeff 方法用于计算插值给定点的曲线的系数。
        这些系数用于表示插值曲线在特定点的形状和位置。
        具体来说，该方法通过求解线性方程组来找到B样条基函数在给定点上的插值系数。
        此方法的作用是根据输入和输出点计算B样条基函数的系数，
        使得这些基函数能够精确插值给定的输入输出点对。
        这样可以用于拟合数据或在模型中应用非线性变换。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_feature
        
        # 确保输出张量的形状正确
        assert y.size() == (x.size(0), self.in_feature, self.out_feature)

        # 计算B样条基函数
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        
        # 转置输出张量
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # 使用线性代数方法求解线性方程组，找到系数
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        
        # 调整结果的形状
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        # 确保结果张量的形状正确
        assert result.size() == (
            self.out_feature,
            self.in_feature,
            self.grid_size + self.spline_order,
        )
        
        # 返回连续存储的结果张量
        return result.contiguous()


class KAN_norm(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
        weight_norm=False,
        init = 'origin'
    ):
        super(KAN_norm, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.weight_norm = weight_norm

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    init=init,

                )
            )
            self.layers.append(
                norm(out_features)
            )
        self.layers.append(
            KANLinear(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                init=init,
            )
        )



    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if layer._get_name() == "KANLinear":
                if self.update_grid and self.train():
                    layer.update_grid(x)
                if self.weight_norm:
                    layer.weight_norm()
            # print(4)
            x = layer(x)
            # print(5)
        # print(6)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

class KAN_normNew(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
        weight_norm=False,
        init = 'origin'
    ):
        super(KAN_normNew, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.weight_norm = weight_norm

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinearNorm(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    init=init,
                    use_norm=norm
                )
            )
        self.layers.append(
            KANLinearNorm(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                init=init,
                use_norm=norm
            )
        )



    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if layer._get_name() == "KANLinear":
                if self.update_grid and self.train():
                    layer.update_grid(x)
                if self.weight_norm:
                    layer.weight_norm()
            # print(4)
            # print(layer)
            # print(x.size())
            x = layer(x)
            # print(x.size())
            # print(5)
        # print(6)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

class KAN_norm_noRes(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
        weight_norm=False
    ):
        super(KAN_norm_noRes, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.weight_norm = weight_norm

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinearNoRes(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
            self.layers.append(
                norm(out_features)
            )
        self.layers.append(
            KANLinearNoRes(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        )



    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if layer._get_name() == "KANLinear":
                if self.update_grid and self.train():
                    layer.update_grid(x)
                if self.weight_norm:
                    layer.weight_norm()
            # print(4)
            x = layer(x)
            # print(5)
        # print(6)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

class KAN_dropout(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
        dropout_rate=0.05
    ):
        super(KAN_dropout, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.dropout_rate = dropout_rate

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
            self.layers.append(
                norm(out_features)
            )
            self.layers.append(nn.Dropout(p=dropout_rate))

        self.layers.append(
            KANLinear(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        )



    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if layer._get_name() == "KANLinear":
                if self.update_grid:
                    layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

def calculate_in_bounds_ratio(outputs):
    """
    计算输出中在 [-1, 1] 的比例
    :param outputs: Tensor，形状为 (batch_size, num_elements)
    :return: 在 [-1, 1] 的比例
    """
    out_of_bounds = torch.logical_and(outputs <= 2.2, outputs >= -2.2)
    ratio = out_of_bounds.float().mean().item()  # 计算比例
    return ratio


class KAN_static(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
    ):
        super(KAN_static, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.depth = len(layers_hidden) -1
        self.ratio_init()
        


        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
            self.layers.append(
                norm(out_features)
            )

        self.layers.append(
            KANLinear(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        i=0
        for layer in self.layers:
            if layer._get_name() == "KANLinear":
                if self.update_grid:
                    layer.update_grid(x)
                ratio = calculate_in_bounds_ratio(x)
                self.in_bounds_ratios[i].append(ratio)
                i += 1
            x = layer(x)

        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
    def ratio_init(self):
        self.in_bounds_ratios = [[] for _ in range(self.depth)]


class KAN_WN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
    ):
        super(KAN_WN, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.depth = len(layers_hidden) -1
        self.ratio_init()
        


        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinear_WN(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
            self.layers.append(
                norm(out_features)
            )

        self.layers.append(
            KANLinear_WN(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        i=0
        for layer in self.layers:
            if layer._get_name() == "KANLinear_WN":
                if self.update_grid:
                    layer.update_grid(x)
                ratio = calculate_in_bounds_ratio(x)
                self.in_bounds_ratios[i].append(ratio)
                i += 1
            x = layer(x)

        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
    def ratio_init(self):
        self.in_bounds_ratios = [[] for _ in range(self.depth)]


class KAN_WN_new(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
    ):
        super(KAN_WN_new, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.depth = len(layers_hidden) -1
        self.ratio_init()
        


        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinear_WN_new(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
            self.layers.append(
                norm(out_features)
            )

        self.layers.append(
            KANLinear_WN_new(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        i=0
        for layer in self.layers:
            if layer._get_name() == "KANLinear_WN_new":
                if self.update_grid:
                    layer.update_grid(x)
                ratio = calculate_in_bounds_ratio(x)
                self.in_bounds_ratios[i].append(ratio)
                i += 1
            x = layer(x)

        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
    def ratio_init(self):
        self.in_bounds_ratios = [[] for _ in range(self.depth)]
