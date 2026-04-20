import os

import numpy as np
import torch


class GeometryTracker:
    def __init__(self, enabled=False, every=1, metrics=None):
        self.enabled = bool(enabled)
        self.every = max(int(every), 1)
        self.metrics = metrics
        self.start_vector = None
        self.prev_vector = None
        self.prev_prev_vector = None
        self.final_vector = None

    def capture_start(self, model):
        if not self.enabled:
            return
        self.start_vector = flatten_parameters(model)
        self.prev_vector = None
        self.prev_prev_vector = None

    def track(self, model, step, epoch=None):
        if not self.enabled or self.start_vector is None:
            return
        if step % self.every != 0:
            return

        current = flatten_parameters(model)
        distance = torch.linalg.vector_norm(current - self.start_vector).item()
        record = {
            "step": int(step),
            "epoch": int(epoch) if epoch is not None else None,
            "distance_from_init": float(distance),
        }

        scalars = {"distance_from_init": distance}
        if self.prev_vector is not None and self.prev_prev_vector is not None:
            current_update = current - self.prev_vector
            previous_update = self.prev_vector - self.prev_prev_vector
            chord = current - self.prev_prev_vector

            current_norm = torch.linalg.vector_norm(current_update).item()
            previous_norm = torch.linalg.vector_norm(previous_update).item()
            chord_norm = torch.linalg.vector_norm(chord).item()
            denom = max(current_norm * previous_norm, 1e-12)
            cosine_similarity = torch.dot(current_update, previous_update).item() / denom
            curve_rate = (current_norm + previous_norm) / max(chord_norm, 1e-12)

            scalars.update(
                {
                    "update_rate": current_norm,
                    "curve_rate": curve_rate,
                    "cosine_similarity": cosine_similarity,
                }
            )
            record.update(scalars)

        if self.metrics is not None:
            self.metrics.append_record("geometry", record)
            self.metrics.log_scalars(
                scalars,
                step=step,
                epoch=epoch,
                prefix="geometry",
            )

        self.prev_prev_vector = self.prev_vector
        self.prev_vector = current
        self.final_vector = current

    def finish(self):
        if not self.enabled or self.metrics is None:
            return
        self.metrics.save_records("geometry", stem="geometry_stats")


def flatten_parameters(model):
    return torch.cat(
        [param.detach().cpu().reshape(-1) for param in model.parameters()]
    ).double()


def assign_parameters(model, theta):
    offset = 0
    theta = theta.detach().cpu()
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            value = theta[offset : offset + numel].view_as(param).to(
                device=param.device,
                dtype=param.dtype,
            )
            param.copy_(value)
            offset += numel


def random_isosceles_right_directions(theta0, thetav, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    theta0 = theta0.detach().cpu().double()
    thetav = thetav.detach().cpu().double()
    h = thetav - theta0
    h_norm = torch.linalg.vector_norm(h)
    if h_norm.item() < 1e-12:
        raise ValueError("theta0 and thetav are too close to define a landscape slice.")

    h_unit = h / h_norm
    r = torch.randn_like(h_unit)
    r = r - torch.dot(r, h_unit) * h_unit
    r_norm = torch.linalg.vector_norm(r)
    if r_norm.item() < 1e-12:
        raise ValueError("Failed to sample a direction orthogonal to the parameter path.")
    r_unit = r / r_norm

    scale = h_norm / np.sqrt(2.0)
    u = scale * (h_unit + r_unit) / np.sqrt(2.0)
    v = scale * (h_unit - r_unit) / np.sqrt(2.0)
    return u, v


def batched_loss(model, loss_fn, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                raise ValueError("Expected each batch to contain at least inputs and targets.")
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("No samples were evaluated for landscape loss computation.")
    return total_loss / total_samples


def loss_landscape_slice(
    model,
    theta0,
    u,
    v,
    loss_fn,
    loader,
    device,
    grid_size=41,
    s_range=(-1.0 / 3.0, 4.0 / 3.0),
    t_range=(-1.0 / 3.0, 4.0 / 3.0),
    max_batches=None,
):
    theta0 = theta0.detach().cpu().double()
    u = u.detach().cpu().double()
    v = v.detach().cpu().double()

    s_values = torch.linspace(float(s_range[0]), float(s_range[1]), steps=int(grid_size))
    t_values = torch.linspace(float(t_range[0]), float(t_range[1]), steps=int(grid_size))
    losses = torch.zeros((len(t_values), len(s_values)), dtype=torch.double)
    original = flatten_parameters(model)

    try:
        for j, t in enumerate(t_values):
            for i, s in enumerate(s_values):
                theta = theta0 + s * u + t * v
                assign_parameters(model, theta)
                losses[j, i] = batched_loss(
                    model,
                    loss_fn,
                    loader,
                    device=device,
                    max_batches=max_batches,
                )
    finally:
        assign_parameters(model, original)

    s_grid, t_grid = np.meshgrid(s_values.numpy(), t_values.numpy())
    return s_grid, t_grid, losses.numpy()


def save_landscape_arrays(output_dir, s_grid, t_grid, losses, stem="landscape"):
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f"{stem}.npz"),
        s_grid=s_grid,
        t_grid=t_grid,
        losses=losses,
    )


def plot_landscape_2d(
    s_grid,
    t_grid,
    losses,
    save_path,
    theta_v=(1.0, 1.0),
    log_scale=True,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    losses_safe = np.clip(losses, 1e-14, None)

    plt.figure(figsize=(8, 7))
    if log_scale:
        contour = plt.contourf(
            s_grid,
            t_grid,
            losses_safe,
            levels=60,
            cmap="viridis",
            norm=LogNorm(vmin=losses_safe.min(), vmax=losses_safe.max()),
            alpha=0.9,
        )
        plt.contour(
            s_grid,
            t_grid,
            losses_safe,
            levels=np.logspace(
                np.log10(losses_safe.min()),
                np.log10(losses_safe.max()),
                15,
            ),
            colors="white",
            alpha=0.3,
            linewidths=0.5,
        )
        colorbar = plt.colorbar(contour, shrink=0.8, format="%.2e")
        colorbar.set_label("Loss (log scale)", fontsize=12)
    else:
        contour = plt.contourf(s_grid, t_grid, losses, levels=60, cmap="viridis", alpha=0.9)
        colorbar = plt.colorbar(contour, shrink=0.8, format="%.2e")
        colorbar.set_label("Loss", fontsize=12)

    plt.scatter(0.0, 0.0, c="red", s=120, marker="o", edgecolors="white", linewidth=2, zorder=5)
    plt.text(0.0, 0.0, "  theta0", color="white", fontsize=12, fontweight="bold", zorder=6)

    if theta_v is not None:
        plt.scatter(
            theta_v[0],
            theta_v[1],
            c="lime",
            s=150,
            marker="*",
            edgecolors="black",
            linewidth=1,
            zorder=5,
        )
        plt.text(
            theta_v[0],
            theta_v[1],
            "  theta_v",
            color="white",
            fontsize=12,
            fontweight="bold",
            zorder=6,
        )
        plt.plot([0.0, theta_v[0]], [0.0, theta_v[1]], "w--", alpha=0.7, linewidth=2, label="u + v")
        plt.legend(loc="upper left")

    plt.xlabel("s (along u)", fontsize=12)
    plt.ylabel("t (along v)", fontsize=12)
    plt.title("Loss Landscape Slice", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_landscape_3d(s_grid, t_grid, losses, save_path, log_scale=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatter, LogLocator

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    losses_safe = np.clip(losses, 1e-8, None)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if log_scale:
        surface = ax.plot_surface(
            s_grid,
            t_grid,
            losses_safe,
            cmap="viridis",
            norm=LogNorm(vmin=losses_safe.min(), vmax=losses_safe.max()),
            linewidth=0,
            antialiased=False,
            alpha=0.9,
        )
        ax.set_zscale("log")
        ax.set_zlim(losses_safe.min(), losses_safe.max())
        ax.zaxis.set_major_locator(LogLocator(base=10.0))
        ax.zaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
        ax.set_zlabel("Loss (log scale)", fontsize=12)
    else:
        surface = ax.plot_surface(
            s_grid,
            t_grid,
            losses,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
            alpha=0.9,
        )
        ax.set_zlabel("Loss", fontsize=12)

    ax.scatter(0.0, 0.0, losses[0, 0], color="red", s=100, label="theta0", depthshade=True)
    ax.set_xlabel("s (along u)", fontsize=12)
    ax.set_ylabel("t (along v)", fontsize=12)
    ax.set_title("3D Loss Landscape", fontsize=14)
    fig.colorbar(surface, shrink=0.6, aspect=20, pad=0.1).set_label("Loss", fontsize=12)
    ax.view_init(elev=30, azim=225)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
