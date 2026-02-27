# %%
import comet_ml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import os

load_dotenv()

# %%
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

# %%
sdf = torch.tensor(np.load("./data/distance_field.npy"), dtype=torch.float).to(device)
uv = torch.tensor(np.load("./data/uv.npy"))
vv = torch.tensor(np.load("./data/vv.npy"))
# plt.imshow(sdf)

# %% [markdown]
# # Here define the model of our neural network

# %%
start_pos = torch.tensor([5, 5]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)
end_pos = torch.tensor([20, 35]).to(
    device
)  # np.random.rand(2) * 40, dtype=torch.float).to(device)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = nn.Linear(1, 64)
        # change activ fun. more smooth
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 64)
        self.dense4 = nn.Sequential(nn.Linear(64, 4))
        self.T = nn.Parameter(torch.tensor([10.0]))

    def forward(self, t):
        x = torch.sin(self.dense1(t))
        x = torch.sin(self.dense2(x))
        x = torch.sin(self.dense3(x))
        x = self.dense4(x)

        return torch.cat(
            [(1 - t) * start_pos + t * end_pos + t * (1 - t) * x[0:2], x[2:]], 0
        )


model = PINN().to(device)
print(model)


# %% [markdown]
# # Next we define the loss function


# %%
class PathLoss(nn.Module):
    def __init__(self):
        super(PathLoss, self).__init__()

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
        softplus_loss = F.softplus(-3 * sdf_vals).sum()
        sdf_loss = (1 / (torch.pow(F.softplus(sdf_vals), 2) + 1e-9)).sum()

        # Physics loss
        v_diff = torch.diff(out[:, 0:2], prepend=start_pos.unsqueeze(0), dim=0) / (
            T / 100
        )
        physics_error = torch.cat(
            [
                v_diff[:, 0] - out[:, 2] * torch.cos(out[:, 3]),
                v_diff[:, 1] - out[:, 2] * torch.sin(out[:, 3]),
            ]
        )
        physics_loss = torch.pow(physics_error, 2).sum()

        if warming:
            return 100 * softplus_loss
        return 100 * softplus_loss + sdf_loss + physics_loss


loss = PathLoss().to(device)

# %% [markdown]
# # Training Function

# %%
hyper_params = {
    "learning_rate": 0.01,
    "steps": 4000,
    "path_steps": 100,
}
optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])


def train(model, optimizer, device, sdf, loss_fn):
    model.train()
    for i in range(hyper_params["steps"]):
        optimizer.zero_grad()
        path_list = []

        for t in range(0, hyper_params["path_steps"]):
            t_tensor = torch.tensor(t * 1e-2, dtype=torch.float32, device=device)
            path_list.append(model(t_tensor.unsqueeze(0)))

        path = torch.stack(path_list)
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
            omega_list = []
            for point in path_list:
                path_x.append(point[0].cpu().detach().numpy())
                path_y.append(point[1].cpu().detach().numpy())
                v_list.append(point[2].cpu().detach().numpy())
                omega_list.append(point[3].cpu().detach().numpy())
            fig, ax = plt.subplots(1, 3)
            ax[0].plot()
            ax[0].plot(path_x, path_y)
            ax[1].plot(v_list)
            ax[1].title("V")
            ax[2].plot(omega_list)
            ax[2].suptitle("Omega")
            ax[0].imshow(sdf.cpu().detach().numpy(), origin="lower")
            ax[0].scatter(*start_pos.cpu().detach().numpy())
            ax[0].scatter(*end_pos.cpu().detach().numpy())
            experiment.log_figure(fig, step=i)


# %% [markdown]
# # Run Training

# %%

experiment.log_parameters(hyper_params)
with experiment.train():
    train(model, optimizer, device, sdf, loss)
experiment.end()
