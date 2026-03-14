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

start_pos = torch.tensor([20, 20, 0, 0]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)
end_pos = torch.tensor([5, 5, 0, 0]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = nn.Linear(1, 64)
        # change activ fun. more smooth
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 64)
        self.dense4 = nn.Sequential(nn.Linear(64, 6))
        self.T = nn.Parameter(torch.tensor([10.0]))

    def forward(self, t):
        x = torch.sin(self.dense1(t))
        x = torch.sin(self.dense2(x))
        x = torch.sin(self.dense3(x))
        x = self.dense4(x)
        path_coords = (
            (1 - t) * start_pos[0:4].view(1, 4)
            + t * end_pos[0:4].view(1, 4)
            + t * (1 - t) * x[:, 0:4]
        )
        return torch.cat([path_coords, x[:, 4:]], dim=1)


model = PINN().to(device)
print(model)


class PathLoss(nn.Module):
    def __init__(self, experiment):
        super(PathLoss, self).__init__()
        self.experiment = experiment
        self.step = 0

    def forward(self, out, sdf, warming, T):
        # path: (100, 2), sdf: (H, W) tensor
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
        sdf_loss = (1 / (torch.pow(F.softplus(sdf_vals), 2) + 1e-9)).sum()

        # Physics loss
        v_diff = torch.diff(out[:, 0:2], prepend=start_pos[0:2].unsqueeze(0), dim=0) / (
            T / 100
        )
        a_diff = torch.diff(out[:, 2:4], prepend=start_pos[2:4].unsqueeze(0), dim=0) / (
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
        a_star_dist = torch.cdist(grid, turning_points)
        a_star_penalty = torch.pow(torch.min(a_star_dist), 2)

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
    t_steps = torch.arange(100, device=device).float().unsqueeze(1) * 0.01
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
            for point in path:
                path_x.append(point[0].cpu().detach().numpy())
                path_y.append(point[1].cpu().detach().numpy())
                v_list.append(point[2].cpu().detach().numpy())
                phi_list.append(point[3].cpu().detach().numpy())
                a_list.append(point[4].cpu().detach().numpy())
                omega_list.append(point[5].cpu().detach().numpy())
            fig, (ax1, ax2) = plt.subplots(2, 2)
            ax1[0].plot()
            ax1[0].plot(path_x, path_y)
            ax2[0].plot(v_list, label="v")
            ax2[0].plot(phi_list, label="phi")
            ax1[1].plot(a_list, label="a")
            ax2[1].plot(omega_list, label="omega")
            ax1[0].imshow(sdf.cpu().detach().numpy(), origin="lower")
            plot_points = turning_points.cpu().detach().numpy()
            ax1[0].scatter(plot_points[:, 0], plot_points[:, 1])
            sp = start_pos.cpu().detach().numpy()
            ax1[0].scatter(sp[0], sp[1], color="red")
            ep = end_pos.cpu().detach().numpy()
            ax1[0].scatter(ep[0], ep[1], color="green")
            ax2[0].legend()
            ax1[1].legend()
            ax2[1].legend()
            experiment.log_figure(fig, step=i)


experiment.log_parameters(hyper_params)
with experiment.train():
    train(model, optimizer, device, sdf, loss)
experiment.end()
