import argparse

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


def add_dataset_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Dataset Options")
    group.add_argument("--num-samples", type=int, default=10000)
    group.add_argument("--train-ratio", type=float, default=0.7)
    group.add_argument("--val-ratio", type=float, default=0.15)
    group.add_argument("--function", default="default")
    group.add_argument("--error", type=float, default=0.05)
    group.add_argument("--curve-points", type=int, default=200)
    group.add_argument("--curve-min", type=float, default=-3.0)
    group.add_argument("--curve-max", type=float, default=3.0)
    return group


class KANDatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        func_name = self._normalize_function_name(self.cfg.function)
        define_fn = getattr(self, f"define_{func_name}", None)
        if define_fn is None:
            raise ValueError(f"Unsupported function setting: {self.cfg.function}")
        define_fn()
        return self.splits

    def build_curve(self, num_points=200):
        num_points = int(num_points)
        x_curve = torch.zeros(num_points, self.cfg.input_dim)
        x_curve[:, 0] = torch.linspace(
            float(getattr(self.cfg, "curve_min", -3.0)),
            float(getattr(self.cfg, "curve_max", 3.0)),
            num_points,
        )
        y_curve = self.target_fn(x_curve).unsqueeze(1)
        return x_curve, y_curve

    def _normalize_function_name(self, func):
        name = str(func).lower()
        mapping = {
            "default": "default",
            "1": "func1",
            "2": "func2",
            "3": "func3",
            "4": "func4",
            "f": "funcf",
            "3r": "func3r",
        }
        return mapping.get(name, name)

    def _sample_inputs(self):
        generator = torch.Generator().manual_seed(self.cfg.seed)
        return torch.randn(self.cfg.num_samples, self.cfg.input_dim, generator=generator)

    def _sample_noise(self, num_samples):
        generator = torch.Generator().manual_seed(self.cfg.seed + 1)
        error = float(self.cfg.error)
        return torch.randn(num_samples, generator=generator) * (error * 2) - error

    def _build_dataset(self, target_fn):
        if self.cfg.input_dim < 3:
            raise ValueError("KAN dataset expects input_dim >= 3.")

        self.target_fn = target_fn
        x_data = self._sample_inputs()
        y_true = self.target_fn(x_data).unsqueeze(1)
        y_noise = self._sample_noise(y_true.shape[0]).unsqueeze(1)
        y_data = y_true + y_noise
        self.splits = self._split_tensors(x_data, y_data, y_true)

    def _split_tensors(self, x_data, y_data, y_true):
        train_size = int(self.cfg.train_ratio * self.cfg.num_samples)
        val_size = int(self.cfg.val_ratio * self.cfg.num_samples)
        test_size = self.cfg.num_samples - train_size - val_size
        if test_size <= 0:
            raise ValueError("Invalid dataset split; test set is empty.")

        train_end = train_size
        val_end = train_size + val_size
        splits = {
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
        splits.update(self._build_loaders(splits))
        return splits

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

    def define_default(self):
        self._build_dataset(lambda x: equation(x[:, :3], "default"))

    def define_func1(self):
        self._build_dataset(lambda x: equation(x[:, :3], 1))

    def define_func2(self):
        self._build_dataset(lambda x: equation(x[:, :3], 2))

    def define_func3(self):
        self._build_dataset(lambda x: equation(x[:, :3], 3))

    def define_func4(self):
        self._build_dataset(lambda x: equation(x[:, :3], 4))

    def define_funcf(self):
        self._build_dataset(lambda x: equation(x[:, :3], "f"))

    def define_func3r(self):
        self._build_dataset(lambda x: equation(x[:, :3], "3r"))


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
    data = KANDatasetBuilder(cfg).build()
    return (
        data["x_train"],
        data["x_val"],
        data["x_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
        data["y_true_test"],
    )
