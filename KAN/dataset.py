import torch
from torch.utils.data import DataLoader, TensorDataset


def equation(x, func="default"):
    a = x[:, 0]
    b = x[:, 1]
    c = x[:, 2]

    if func == 1:
        return torch.cos(2 * torch.pi / (1 - a**2 - b**2 - c**2))
    if func == 2:
        return 1 / (a + b + c)
    if func == 3:
        return a**2 + b**2 + c**2
    if func == 4:
        return 1 / (1 + a**2 + b**2 + c**2)
    if func == "f":
        return (a + b + c) / (1 + a * b + a * c + b * c)
    if func == "3r":
        return a**3 + b**3 + c**3

    return (
        torch.sin(2 * a)
        + 0.5 * torch.cos(3 * b)
        - 1.2 * b**2
        + 3 * a * b
        - 2.5 * torch.exp(-c)
        + b * c
        + 2.5
    )


class RegressionDatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        x_data = self._sample_inputs()
        y_true = self._target_function(x_data).unsqueeze(1)
        y_noise = self._sample_noise(y_true.shape[0]).unsqueeze(1)
        y_data = y_true + y_noise

        splits = self._split_tensors(x_data, y_data, y_true)
        loaders = self._build_loaders(splits)
        return {**splits, **loaders}

    def build_curve(self, num_points=200):
        num_points = int(num_points)
        x_curve = torch.zeros(num_points, self.cfg.input_dim)
        x_curve[:, 0] = torch.linspace(
            float(getattr(self.cfg, "curve_min", -3.0)),
            float(getattr(self.cfg, "curve_max", 3.0)),
            num_points,
        )
        y_curve = self._target_function(x_curve).unsqueeze(1)
        return x_curve, y_curve

    def _sample_inputs(self):
        generator = torch.Generator().manual_seed(self.cfg.seed)
        return torch.randn(self.cfg.num_samples, self.cfg.input_dim, generator=generator)

    def _sample_noise(self, num_samples):
        generator = torch.Generator().manual_seed(self.cfg.seed + 1)
        error = float(self.cfg.error)
        return torch.randn(num_samples, generator=generator) * (error * 2) - error

    def _target_function(self, x):
        if self.cfg.input_dim < 3:
            raise ValueError("RegressionDatasetBuilder expects input_dim >= 3.")
        return equation(x[:, :3], self.cfg.function)

    def _split_tensors(self, x_data, y_data, y_true):
        train_size = int(self.cfg.train_ratio * self.cfg.num_samples)
        val_size = int(self.cfg.val_ratio * self.cfg.num_samples)
        test_size = self.cfg.num_samples - train_size - val_size
        if test_size <= 0:
            raise ValueError("Invalid dataset split; test set is empty.")

        train_end = train_size
        val_end = train_size + val_size
        return {
            "x_train": x_data[:train_end],
            "y_train": y_data[:train_end],
            "y_true_train": y_true[:train_end],
            "x_val": x_data[train_end:val_end],
            "y_val": y_data[train_end:val_end],
            "y_true_val": y_true[train_end:val_end],
            "x_test": x_data[val_end:],
            "y_test": y_data[val_end:],
            "y_true_test": y_true[val_end:],
        }

    def _build_loaders(self, splits):
        train_dataset = TensorDataset(splits["x_train"], splits["y_train"])
        val_dataset = TensorDataset(splits["x_val"], splits["y_val"])
        test_dataset = TensorDataset(splits["x_test"], splits["y_test"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size[0],
            shuffle=True,
        )
        eval_batch_size = self.cfg.batch_size[1] if len(self.cfg.batch_size) > 1 else self.cfg.batch_size[0]
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }


def prepare_data(seed, error=0.05, func="default", num_samples=10000):
    cfg = type(
        "_Cfg",
        (),
        {
            "input_dim": 3,
            "output_dim": 1,
            "batch_size": [num_samples, num_samples],
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "function": func,
            "num_samples": num_samples,
            "error": error,
            "seed": seed,
        },
    )()
    data = RegressionDatasetBuilder(cfg).build()
    return (
        data["x_train"],
        data["x_val"],
        data["x_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
        data["y_true_test"],
    )
