# %%
import comet_ml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import os
import random

from map_generation import Point, Rectangle, UShape, SShape
from map_generation import create_u_shape, create_s_shape
from experiment_logger import Logger, Metrics


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

load_dotenv()

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

comet_api_key = os.getenv("COMET_API_KEY")
comet_project_name = os.getenv("COMET_PROJECT_NAME")
comet_workspace = os.getenv("COMET_WORKSPACE")
experiment = comet_ml.start(
    api_key=comet_api_key, project_name=comet_project_name, workspace=comet_workspace
)
logger = Logger(experiment)

sdf = torch.tensor(np.load("./data/distance_field.npy"), dtype=torch.float).to(device)
uv = torch.tensor(np.load("./data/uv.npy"))
vv = torch.tensor(np.load("./data/vv.npy"))
turning_points = torch.tensor(
    np.load("./data/turning_points.npy"), dtype=torch.float
).to(device)
# plt.imshow(sdf)

# (x, y, v, theta)
start_pos = torch.tensor([40, 50, 0]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)
end_pos = torch.tensor([60, 50, 0]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        upper1 = np.sqrt(6)
        lower1 = -np.sqrt(6)
        upper = np.sqrt(6 / 128)
        lower = -np.sqrt(6 / 128)
        self.dense1 = nn.Linear(1, 128)
        torch.nn.init.uniform_(self.dense1.weight, lower1 * 30, upper1 * 30)
        self.dense2 = nn.Linear(128, 128)
        torch.nn.init.uniform_(self.dense2.weight, lower, upper)
        self.dense3 = nn.Linear(128, 128)
        torch.nn.init.uniform_(self.dense3.weight, lower, upper)
        self.dense4 = nn.Linear(128, 6)
        torch.nn.init.uniform_(self.dense4.weight, lower, upper)
        self.T = nn.Parameter(torch.tensor([10.0]))

    def forward(self, t):
        # None for no bound
        V_MAX, V_MIN = 10, -10
        A_MAX, A_MIN = 5, -5
        OMEGA_MAX, OMEGA_MIN = 1, -1
        t = t.view(-1, 1)  # ensure t is (N, 1)
        x = torch.sin(self.dense1(t))
        x = torch.sin(self.dense2(x))
        x = torch.sin(self.dense3(x))
        x = self.dense4(x)
        # x: (N, 6) = N * (x, y, v, theta, a, omega)
        return torch.cat(
            [
                x[:, 0:2],  # x,y unclamped
                torch.clamp(x[:, 2], V_MIN, V_MAX).unsqueeze(1),  # v clamped
                x[:, 3].unsqueeze(1),  # theta unclamped
                torch.clamp(x[:, 4], A_MIN, A_MAX).unsqueeze(1),  # a clamped
                torch.clamp(x[:, 5], OMEGA_MIN, OMEGA_MAX).unsqueeze(1),  # omega clamped
            ],
            dim=1,
        )  # (N, 6) = N * (x, y, v, theta, a, omega)


model = PINN().to(device)
print(model)

a_star_min_point = None
u = create_u_shape(Point(10, 10))
s = create_s_shape(Point(50, 50))


class PathLoss(nn.Module):
    def __init__(self, logger: Logger, rectangles: UShape | SShape, start: torch.Tensor, end: torch.Tensor):
        super(PathLoss, self).__init__()
        self.logger = logger
        self.step = 0
        self.rectangle_list = rectangles
        self.start = start
        self.end = end

    def forward(self, out, sdf, warming, T, t_steps, iteration):

        # SDF loss
        tau = 1.0
        path_xy = out[:, 0:2]  # (N, 2)
        dists = torch.stack(
            [
                self.distance_from_rect(rect, path_xy)
                for rect in self.rectangle_list.rectangles
            ],
            dim=1,
        )  # (N, 3)
        d_lse = -torch.logsumexp(-tau * dists, dim=1) / tau  # (N,)
        margin = 5
        sdf_loss = torch.clamp(margin - d_lse, min=0).pow(2).mean()

        # Physics loss
        x_dot = (
            torch.autograd.grad(
                out[:, 0],
                t_steps,
                grad_outputs=torch.ones_like(out[:, 0]),
                create_graph=True,
            )[0]
            / T
        )
        y_dot = (
            torch.autograd.grad(
                out[:, 1],
                t_steps,
                grad_outputs=torch.ones_like(out[:, 1]),
                create_graph=True,
            )[0]
            / T
        )
        v_dot = (
            torch.autograd.grad(
                out[:, 2],
                t_steps,
                grad_outputs=torch.ones_like(out[:, 2]),
                create_graph=True,
            )[0]
            / T
        )

        theta_dot = (
            torch.autograd.grad(
                out[:, 3],
                t_steps,
                grad_outputs=torch.ones_like(out[:, 3]),
                create_graph=True,
            )[0]
            / T
        )

        physics_error_x = x_dot - out[:, 2] * torch.cos(out[:, 3])
        physics_error_y = y_dot - out[:, 2] * torch.sin(out[:, 3])
        physics_error_v = v_dot - out[:, 4]
        physics_error_theta = theta_dot - out[:, 5]

        physics_loss = (
            physics_error_x**2
            + physics_error_y**2
            + physics_error_v**2
            + physics_error_theta**2
        ).mean()

        # Optimal path loss
        physical_t = t_steps * T
        dt = torch.diff(physical_t)
        # dt = T / 100
        optimality_loss = (torch.pow(out[:-1, 5], 2) * dt).sum()

        # A* Loss
        # grid: (N, 2), turning_points: (num_turning_points, 2)
        cdist_input_grid = out[:, 0:2].clone().unsqueeze(0)  # (1, N, 2)
        cdist_input_turning = turning_points.unsqueeze(0)  # (1, num_turning_points, 2)
        a_star_dist = torch.cdist(cdist_input_grid, turning_points).squeeze(0)
        # a_star_dist: (N, num_turning_points)
        # print(cdist_input_grid)
        # print(a_star_dist)
        # print(turning_points)
        global a_star_min_point
        min_dist, min_point = torch.min(a_star_dist, dim=0)
        min_dist = min_dist.sum()
        a_star_min_point = min_point[0].cpu().detach().numpy().item()
        a_star_loss = torch.pow(min_dist, 2)

        # Boundary loss
        boundary_loss = (
            (out[0, 0:3] - self.start[0:3]).pow(2).sum()
            + (out[-1, 0:3] - self.end[0:3]).pow(2).sum()
        )

        # Loss coef
        softplus_coef = 100
        sdf_coef = 10
        physics_coef = 1
        optimality_coef = 5
        a_star_coef = 0.05
        boundary_coef = 10
        warming_coef = torch.sigmoid(
            torch.tensor((iteration - 3000) / 100, dtype=torch.float32)
        )

        final_sdf_loss = sdf_coef * sdf_loss
        final_a_star_loss = a_star_coef * a_star_loss
        final_physics_loss = physics_coef * physics_loss * warming_coef
        final_optimality_loss = optimality_coef * optimality_loss * warming_coef
        final_boundary_loss = boundary_coef * boundary_loss

        self.step += 1
        if self.step % 10 == 0:
            metrics = Metrics(
                final_sdf_loss.item(),
                final_a_star_loss.item(),
                final_physics_loss.item(),
                final_optimality_loss.item(),
                warming_coef.item(),
                self.step,
            )
            logger.log_metrics(metrics)

        return (
            # softplus_coef * softplus_loss
            final_sdf_loss
            + final_a_star_loss
            + final_physics_loss
            + final_optimality_loss
            + final_boundary_loss
        )

    def distance_from_rect(self, rect: Rectangle, path: torch.Tensor) -> torch.Tensor:
        x_distance = torch.abs(path[:, 0] - rect.center.x) - rect.width / 2
        y_distance = torch.abs(path[:, 1] - rect.center.y) - rect.height / 2
        outside_distance = torch.sqrt(
            torch.clamp(x_distance, min=0) ** 2 + torch.clamp(y_distance, min=0) ** 2
        )
        inside_distance = torch.clamp(torch.maximum(x_distance, y_distance), max=0)

        return outside_distance + inside_distance

    def calculate_lse_distance(
        self, u: UShape, point: torch.Tensor, tau: float = 1.0
    ) -> torch.Tensor:
        point_2d = point.unsqueeze(0)  # (1, 2) so distance_from_rect works
        distances = torch.stack(
            [
                self.distance_from_rect(u.rectangles[0], point_2d),
                self.distance_from_rect(u.rectangles[1], point_2d),
                self.distance_from_rect(u.rectangles[2], point_2d),
            ]
        )  # (3,)
        lse = -torch.logsumexp(-tau * distances, dim=0) / tau
        return lse

    def in_ushape(self, u: UShape, point: torch.Tensor) -> bool:
        px, py = point[0].item(), point[1].item()
        for rect in u.rectangles:
            if rect.min_coord.x <= px <= rect.max_coord.x:
                if rect.min_coord.y <= py <= rect.max_coord.y:
                    return True
        return False


loss = PathLoss(logger, s, start_pos, end_pos).to(device)

hyper_params = {
    "learning_rate": 0.002,
    "steps": 10000,
    "path_steps": 100,
}
optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])


def train(model, optimizer, device, sdf, loss_fn):
    model.train()

    for i in range(hyper_params["steps"]):
        t_interior = torch.rand(98)
        t_steps = torch.cat([torch.zeros(1), t_interior, torch.ones(1)]).sort()[0]
        t_steps = t_steps.detach().requires_grad_(True).to(device)
        optimizer.zero_grad()
        path = model(t_steps)

        if i < hyper_params["steps"] / 3:
            loss = loss_fn(path, sdf, True, model.T, t_steps, i)
        else:
            loss = loss_fn(path, sdf, False, model.T, t_steps, i)
        loss.backward()
        optimizer.step()
        if i % 250 == 0:
            path_np = path.detach().cpu().numpy()
            sdf_fig = sdf.detach().cpu().numpy()
            plot_points = turning_points.detach().cpu().numpy()
            sp = start_pos.detach().cpu().numpy()
            ep = end_pos.detach().cpu().numpy()
            logger.log_figure(
                loss=loss.item(),
                output=path_np,
                sdf=sdf_fig,
                turning_points=plot_points,
                start_pos=sp,
                end_pos=ep,
                step=i,
                a_star_point=a_star_min_point,
            )


logger.log_parameters(hyper_params)
with logger.get_experiment().train():
    train(model, optimizer, device, sdf, loss)
logger.end_experiment()
