import numpy as np
import deepxde as dde
import torch


class PDEBuilder:
    def __init__(self, cfg, model, optimizer):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer

    def build(self):
        if self.cfg.pde_type == 'poisson':
            self.define_poisson()
        elif self.cfg.pde_type == 'helmholtz':
            self.define_helmholtz()
        elif self.cfg.pde_type == 'helmholtz_learnable':
            self.define_helmholtz_learnable()
        elif self.cfg.pde_type == 'helmholtz2d':
            self.define_helmholtz2d()
        elif self.cfg.pde_type == 'wave':
            self.define_wave()
        elif self.cfg.pde_type == 'klein_gordon':
            self.define_klein_gordon()
        elif self.cfg.pde_type == 'convdiff':
            self.define_convdiff()
        elif self.cfg.pde_type == 'cavity':
            self.define_cavity()
        elif self.cfg.pde_type == 'allen_cahn':
            self.define_allen_cahn()
        else:
            raise ValueError("Unsupported PDE type")
        return self.data, self.net, self.model

    def _as_col(self, z):
        # ensure (N,1)
        return z if z.ndim == 2 else z.reshape(-1, 1)

    def _is_torch(self, x):
        return torch.is_tensor(x)


    def define_poisson(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x)
            return -dy_xx - 1

        def boundary(x, on_boundary):
            return on_boundary

        def func(x):
            if torch.is_tensor(x):
                x0 = x[:, 0:1]
                return (x0**2 - 1.0) / 2.0
            else:
                x0 = x[:, 0:1]
                return (x0**2 - 1.0) / 2.0


        geom = dde.geometry.Interval(-1, 1)
        bc = dde.DirichletBC(geom, func, boundary)
        self.data = dde.data.PDE(geom, pde, bc, num_domain=1000, num_boundary=200, solution=func)
        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics)

    def define_helmholtz(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            pi = torch.pi
            rhs = 2.0 * (pi**2) * torch.sin(pi * x[:, 0:1])
            return -dy_xx + (pi**2) * y - rhs

        def boundary(x, on_boundary):
            return on_boundary

        def func(x):
            if torch.is_tensor(x):
                return torch.sin(torch.pi * x[:, 0:1])
            else:
                return np.sin(np.pi * x[:, 0:1])

        geom = dde.geometry.Interval(-1, 1)
        bc = dde.DirichletBC(geom, func, boundary)
        self.data = dde.data.PDE(geom, pde, bc, num_domain=5000, num_boundary=500, solution=func)
        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics)

    def define_helmholtz_learnable(self):
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            pi = torch.pi
            rhs = 2.0 * (pi**2) * torch.sin(pi * x[:, 0:1])
            return -dy_xx + (pi**2) * y - rhs

        def boundary(x, on_boundary):
            return on_boundary

        def func(x):
            if torch.is_tensor(x):
                return torch.sin(torch.pi * x[:, 0:1])
            else:
                return np.sin(np.pi * x[:, 0:1])

        geom = dde.geometry.Interval(-1, 1)
        bc = dde.DirichletBC(geom, func, boundary)
        self.data = dde.data.PDE(geom, pde, bc, num_domain=5000, num_boundary=500, solution=func)
        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics, loss_weights=[self.cfg.loss_weights[0], 1.0])

    def define_helmholtz2d(self):
        geom = dde.geometry.Rectangle([-1, -1], [1, 1])
        alpha = (1 * np.pi) ** 2 + (4 * np.pi) ** 2

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
            return u_xx + u_yy + alpha * u

        bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda x, on_b: on_b)

        def exact_u(x):
            return np.sin(np.pi * x[:, 0:1]) * np.sin(4 * np.pi * x[:, 1:2])

        self.data = dde.data.PDE(
            geom,
            pde,
            bc,
            num_domain=30000,
            num_boundary=4000,
            solution=exact_u,
            train_distribution="uniform",
        )

        loss_weights = [1.0, 10.0]

        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics, loss_weights=loss_weights)

    def define_wave(self):
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        def pde(x, u):
            u_tt = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_xx = dde.grad.hessian(u, x, component=0, i=1, j=1)
            return u_tt - 4.0 * u_xx

        def on_left(x, on_b):
            return on_b and np.isclose(x[1], 0.0)

        def on_right(x, on_b):
            return on_b and np.isclose(x[1], 1.0)

        bc0 = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_left)
        bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_right)

        def ic_u(x):
            return np.sin(np.pi * x[:, 1:2]) + 0.5 * np.sin(4 * np.pi * x[:, 1:2])

        ic0 = dde.icbc.IC(geomtime, ic_u, lambda x, on_i: on_i)

        def ic_ut(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=0)
            return u_t

        ic1 = dde.icbc.OperatorBC(geomtime, ic_ut, lambda x, on_i: on_i)

        self.data = dde.data.TimePDE(
            geomtime,
            pde,
            [bc0, bc1, ic0, ic1],
            num_domain=20000,
            num_boundary=2000,
            num_initial=2000,
            train_distribution="uniform",
        )

        loss_weights = [1.0, 100.0, 100.0, 100.0, 100.0]

        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics, loss_weights=loss_weights)

    def define_klein_gordon(self):
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        def pde(x, u):
            u_tt = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_xx = dde.grad.hessian(u, x, component=0, i=1, j=1)
            return u_tt - u_xx + u**3

        def on_left(x, on_b):
            return on_b and np.isclose(x[1], 0.0)

        def on_right(x, on_b):
            return on_b and np.isclose(x[1], 1.0)

        bc0 = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_left)
        bc1 = dde.icbc.DirichletBC(
            geomtime,
            lambda x: np.cos(5 * np.pi * x[:, 0:1]) + x[:, 0:1] ** 3,
            on_right,
        )

        ic0 = dde.icbc.IC(geomtime, lambda x: x[:, 1:2], lambda x, on_i: on_i)

        def ic_ut(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=0)
            return u_t

        ic1 = dde.icbc.OperatorBC(geomtime, ic_ut, lambda x, on_i: on_i)

        self.data = dde.data.TimePDE(
            geomtime,
            pde,
            [bc0, bc1, ic0, ic1],
            num_domain=20000,
            num_boundary=2000,
            num_initial=2000,
            train_distribution="uniform",
        )

        loss_weights = [1.0, 50.0, 50.0, 50.0, 50.0]

        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics, loss_weights=loss_weights)

    def define_convdiff(self):
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        c1, c2, D = 1.0, 1.0, 0.01

        def exact_u(x):
            t = x[:, 0:1]
            xx = x[:, 1:2]
            yy = x[:, 2:3]
            g = np.exp(-100.0 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2))
            return g * np.exp(-t)

        def pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=0)
            u_x = dde.grad.jacobian(u, x, i=0, j=1)
            u_y = dde.grad.jacobian(u, x, i=0, j=2)
            u_xx = dde.grad.hessian(u, x, component=0, i=1, j=1)
            u_yy = dde.grad.hessian(u, x, component=0, i=2, j=2)
            return u_t + c1 * u_x + c2 * u_y - D * (u_xx + u_yy)

        bc = dde.icbc.DirichletBC(geomtime, exact_u, lambda x, on_b: on_b)

        def ic_u(x):
            xx = x[:, 1:2]
            yy = x[:, 2:3]
            return np.exp(-100.0 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2))

        ic = dde.icbc.IC(geomtime, ic_u, lambda x, on_i: on_i)

        self.data = dde.data.TimePDE(
            geomtime,
            pde,
            [bc, ic],
            num_domain=30000,
            num_boundary=4000,
            num_initial=4000,
            solution=exact_u,
            train_distribution="uniform",
        )

        loss_weights = [1.0, 10.0, 10.0]

        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics, loss_weights=loss_weights)

    def define_cavity(self):
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        rho = 1056.0
        mu = 0.01

        def pde(x, y):
            u = y[:, 0:1]
            v = y[:, 1:2]
            p = y[:, 2:3]

            u_t = dde.grad.jacobian(y, x, i=0, j=0)
            v_t = dde.grad.jacobian(y, x, i=1, j=0)

            u_x = dde.grad.jacobian(y, x, i=0, j=1)
            u_y = dde.grad.jacobian(y, x, i=0, j=2)
            v_x = dde.grad.jacobian(y, x, i=1, j=1)
            v_y = dde.grad.jacobian(y, x, i=1, j=2)

            p_x = dde.grad.jacobian(y, x, i=2, j=1)
            p_y = dde.grad.jacobian(y, x, i=2, j=2)

            u_xx = dde.grad.hessian(y, x, component=0, i=1, j=1)
            u_yy = dde.grad.hessian(y, x, component=0, i=2, j=2)
            v_xx = dde.grad.hessian(y, x, component=1, i=1, j=1)
            v_yy = dde.grad.hessian(y, x, component=1, i=2, j=2)

            ru = u_t + u * u_x + v * u_y + (1.0 / rho) * p_x - mu * (u_xx + u_yy)
            rv = v_t + u * v_x + v * v_y + (1.0 / rho) * p_y - mu * (v_xx + v_yy)
            rc = u_x + v_y
            return [ru, rv, rc]

        def on_top(x, on_b):
            return on_b and np.isclose(x[2], 1.0)

        def on_bottom(x, on_b):
            return on_b and np.isclose(x[2], 0.0)

        def on_left(x, on_b):
            return on_b and np.isclose(x[1], 0.0)

        def on_right(x, on_b):
            return on_b and np.isclose(x[1], 1.0)

        bc_u_top = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, on_top, component=0)
        bc_v_top = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_top, component=1)

        bc_u_bottom = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_bottom, component=0)
        bc_v_bottom = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_bottom, component=1)
        bc_u_left = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_left, component=0)
        bc_v_left = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_left, component=1)
        bc_u_right = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_right, component=0)
        bc_v_right = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, on_right, component=1)

        ic_u = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_i: on_i, component=0)
        ic_v = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_i: on_i, component=1)
        ic_p = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_i: on_i, component=2)

        self.data = dde.data.TimePDE(
            geomtime,
            pde,
            [
                bc_u_top,
                bc_v_top,
                bc_u_bottom,
                bc_v_bottom,
                bc_u_left,
                bc_v_left,
                bc_u_right,
                bc_v_right,
                ic_u,
                ic_v,
                ic_p,
            ],
            num_domain=50000,
            num_boundary=8000,
            num_initial=8000,
            train_distribution="uniform",
        )

        loss_weights = [0.1, 0.1, 0.1] + [2.0] * 8 + [4.0] * 3

        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics, loss_weights=loss_weights)

    def define_allen_cahn(self):
        epsilon = 0.01

        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x)
            return epsilon * dy_xx + y - y**3

        def boundary(x, on_boundary):
            return on_boundary

        def func(x):
            if torch.is_tensor(x):
                eps = torch.tensor(2.0 * epsilon, dtype=x.dtype, device=x.device).sqrt()
                return torch.tanh(x[:, 0:1] / eps)
            else:
                eps_np = np.sqrt(2.0 * epsilon)
                return np.tanh(x[:, 0:1] / eps_np)

        geom = dde.geometry.Interval(-1, 1)
        bc = dde.DirichletBC(geom, func, boundary)
        self.data = dde.data.PDE(geom, pde, bc, num_domain=10000, num_boundary=500, solution=func)
        self.model.regularizer = None
        self.net = self.model
        self.model = dde.Model(self.data, self.net)
        self.model.compile(optimizer=self.optimizer, metrics=self.cfg.metrics)
