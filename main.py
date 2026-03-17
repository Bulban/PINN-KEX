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

sdf = torch.tensor(np.load("./data/distance_field.npy"), dtype=torch.float).to(device)
uv = torch.tensor(np.load("./data/uv.npy"))
vv = torch.tensor(np.load("./data/vv.npy"))
turning_points = torch.tensor(
    np.load("./data/turning_points.npy"), dtype=torch.float
).to(device)
# plt.imshow(sdf)

# (x, y, v, phi)
start_pos = torch.tensor([20, 20, 0]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)
end_pos = torch.tensor([5, 5, 0]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = nn.Linear(1, 128)
        # change activ fun. more smooth
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 128)
        self.dense4 = nn.Linear(128, 6)
        self.T = nn.Parameter(torch.tensor([10.0]))

    def forward(self, t):
        t = t.view(-1, 1)  # ensure t is (N, 1)
        x = torch.sin(self.dense1(t))
        x = torch.sin(self.dense2(x))
        x = torch.sin(self.dense3(x))
        x = self.dense4(x)
        # x: (N, 6) = N * (x, y, v, phi, a, omega)
        path_coords = (
            (1 - t) * start_pos[0:3].view(1, 3)
            + t * end_pos[0:3].view(1, 3)
            + t * (1 - t) * x[:, 0:3]
        )
        # path_coords: (N, 3) = N * (x, y, v)
        #
        return torch.cat(
            [path_coords, x[:, 3:]], dim=1
        )  # (N, 6) = N * (path_coords.., a, omega)


model = PINN().to(device)
print(model)

a_star_min_point = None
class PathLoss(nn.Module):
    def __init__(self, experiment):
        super(PathLoss, self).__init__()
        self.experiment = experiment
        self.step = 0

    def forward(self, out, sdf, warming, T):
        # out: (N, 6), sdf: (H, W) tensor
        H, W = sdf.shape

        # normalize path coords to [-1, 1] as required by grid_sample
        grid = out[:, 0:2].clone()
        grid[:, 0] = (grid[:, 0] / (W - 1)) * 2 - 1  # x
        grid[:, 1] = (grid[:, 1] / (H - 1)) * 2 - 1  # y

        # grid_sample expects (N, C, H, W) and grid (N, H, W, 2)
        sdf_input = sdf.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        grid_input = grid[:, [1, 0]].unsqueeze(0).unsqueeze(0)  # (1, 1, 100, 2)

        sdf_vals = F.grid_sample(
            sdf_input, grid_input, align_corners=True, padding_mode="zeros"
        )  # (1, 1, 1, 100)
        sdf_vals = sdf_vals.squeeze()
        softplus_loss = F.softplus(-10 * sdf_vals).sum()
        sdf_loss = (1 / (torch.pow(sdf_vals, 2) + 1e-2)).sum()

        # Physics loss
        v_diff = torch.diff(out[:, 0:2], prepend=out[0, 0:2].unsqueeze(0), dim=0) / (
            T / 100
        )
        a_diff = torch.diff(out[:, 2:4], prepend=out[0, 2:4].unsqueeze(0), dim=0) / (
            T / 100
        )
        physics_error = torch.cat(
            [
                v_diff[:, 0] - out[:, 2] * torch.cos(out[:, 3]),
                v_diff[:, 1] - out[:, 2] * torch.sin(out[:, 3]),
                a_diff[:, 0] - out[:, 4],
                a_diff[:, 1] - out[:, 5],
            ]
        )
        physics_loss = torch.pow(physics_error, 2).sum()

        # Optimal path loss
        dt = T / 100
        optimality_loss = torch.pow(out[:, 5] * dt, 2).sum()

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
        a_star_min_point = min_point[0].cpu().detach().numpy()
        a_star_penalty = torch.pow(min_dist, 2)

        # Loss coef
        softplus_coef = 100
        sdf_coef = 1
        physics_coef = 1
        optimality_coef = 100
        a_star_coef = 100

        self.experiment.log_metrics(
            {
                "loss/softplus": softplus_coef * softplus_loss.item(),
                "loss/sdf": sdf_coef * sdf_loss.item(),
                "loss/a_star": a_star_coef * a_star_penalty.item(),
                "loss/physics": physics_coef * physics_loss.item(),
                "loss/optimality": optimality_coef * optimality_loss.item(),
            },
            step=self.step,
        )
        self.step += 1
        if self.step % 10 == 0:
            self.experiment.log_metrics(
                {
                    "loss/softplus": softplus_coef * softplus_loss.item(),
                    "loss/sdf": sdf_coef * sdf_loss.item(),
                    "loss/a_star": a_star_coef * a_star_penalty.item(),
                    "loss/physics": physics_coef * physics_loss.item(),
                    "loss/optimality": optimality_coef * optimality_loss.item(),
                },
                step=self.step,
            )

        if warming:
            return softplus_coef * softplus_loss + a_star_coef * a_star_penalty
        return (
            softplus_coef * softplus_loss
            + sdf_coef * sdf_loss
            + physics_coef * physics_loss
            + optimality_coef * optimality_loss
            + a_star_coef * a_star_penalty
        )


loss = PathLoss(experiment).to(device)

hyper_params = {
    "learning_rate": 0.01,
    "steps": 4000,
    "path_steps": 100,
}
optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])


def train(model, optimizer, device, sdf, loss_fn):
    model.train()
    t_steps = torch.linspace(0, 1, 100, device=device)
    for i in range(hyper_params["steps"]):
        optimizer.zero_grad()
        path = model(t_steps)

        if i < hyper_params["steps"] / 3:
            loss = loss_fn(path, sdf, True, model.T)
        else:
            loss = loss_fn(path, sdf, False, model.T)
        loss.backward()
        optimizer.step()
        if i % 250 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")
            path_x = []
            path_y = []
            v_list = []
            phi_list = []
            a_list = []
            omega_list = []
            path_np = path.cpu().detach().numpy()
            path_x = path_np[:, 0]
            path_y = path_np[:, 1]
            v_list = path_np[:, 2]
            phi_list = path_np[:, 3]
            a_list = path_np[:, 4]
            omega_list = path_np[:, 5]
            fig, (ax1, ax2) = plt.subplots(2, 2)
            ax1[0].plot()
            ax1[0].plot(path_x, path_y)
            ax2[0].plot(v_list, label="v")
            ax2[0].plot(phi_list, label="phi")
            ax1[1].plot(a_list, label="a")
            ax2[1].plot(omega_list, label="omega")
            ax1[0].imshow(sdf.cpu().detach().numpy(), origin="lower")
            plot_points = turning_points.cpu().detach().numpy()
            ax1[0].scatter(
                plot_points[:, 0], plot_points[:, 1], color="magenta", marker="*"
            )
            sp = start_pos.cpu().detach().numpy()
            ax1[0].scatter(sp[0], sp[1], color="limegreen", marker="o")
            ep = end_pos.cpu().detach().numpy()
            ax1[0].scatter(ep[0], ep[1], color="red", marker="x")
            # Plot the point the A-star loss is based on, i.e. the closest point on the path
            ax1[0].scatter(path_x[a_star_min_point.item()], path_y[a_star_min_point.item()], color="yellow", marker="1")
            ax2[0].legend()
            ax1[1].legend()
            ax2[1].legend()
            experiment.log_figure(fig, step=i)
            plt.close()


experiment.log_parameters(hyper_params)
with experiment.train():
    train(model, optimizer, device, sdf, loss)
experiment.end()
