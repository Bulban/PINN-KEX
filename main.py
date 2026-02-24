# %%
import comet_ml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from dotenv import load_dotenv
import os
load_dotenv()

# %%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

comet_api_key = os.getenv("COMET_API_KEY")
comet_project_name = os.getenv("COMET_PROJECT_NAME")
comet_workspace = os.getenv("COMET_WORKSPACE")
experiment = comet_ml.start(api_key=comet_api_key, project_name=comet_project_name, workspace=comet_workspace)

# %%
sdf = torch.tensor(np.load("./data/distance_field.npy"), dtype=torch.float).to(device)
uv = torch.tensor(np.load("./data/uv.npy"))
vv = torch.tensor(np.load("./data/vv.npy"))
#plt.imshow(sdf)

# %% [markdown]
# # Here define the model of our neural network

# %%
start_pos = torch.tensor([5,5]).to(device)#np.random.rand(2) * 40, dtype=torch.float).to(device)
end_pos = torch.tensor([20,35]).to(device)#np.random.rand(2) * 40, dtype=torch.float).to(device)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = nn.Linear(1, 64)
        # change activ fun. more smooth
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 64)
        self.dense4 = nn.Sequential(
            nn.Linear(64, 2))


    def forward(self, t):
        x = torch.sin(self.dense1(t))
        x = torch.sin(self.dense2(x))
        x = torch.sin(self.dense3(x))
        x = self.dense4(x)
        return (1-t)*start_pos + t*end_pos + t * (1-t)*x*40

model = PINN().to(device)
print(model)


# %% [markdown]
# # Next we define the loss function

# %%
class PathLoss(nn.Module):
    def __init__(self):
        super(PathLoss, self).__init__()

    def forward(self, path, sdf, warming):
        # path: (100, 2), sdf: (H, W) tensor
        H, W = sdf.shape

        # normalize path coords to [-1, 1] as required by grid_sample
        grid = path.clone()
        grid[:, 0] = (grid[:, 0] / (W - 1)) * 2 - 1  # x
        grid[:, 1] = (grid[:, 1] / (H - 1)) * 2 - 1  # y

        # grid_sample expects (N, C, H, W) and grid (N, H, W, 2)
        sdf_input = sdf.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)
        grid_input = grid[:, [1, 0]].unsqueeze(0).unsqueeze(0)        # (1, 1, 100, 2)

        sdf_vals = F.grid_sample(sdf_input, grid_input,
                                  align_corners=True,
                                  padding_mode='zeros')  # (1, 1, 1, 100)
        sdf_vals = sdf_vals.squeeze()
        softplus_loss = F.softplus(-3 * sdf_vals).sum()
        sdf_loss = (1 / (torch.pow(F.softplus(sdf_vals),2) + 1e-9)).sum()
        if warming:
            return 100*softplus_loss
        return 100*softplus_loss + sdf_loss
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
            loss = loss_fn(path, sdf, True)
        else:
            loss = loss_fn(path,sdf, False)
        loss.backward()
        optimizer.step()
        if i % 250 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")
            path_x = []
            path_y = []
            for point in path_list:
                path_x.append(point[0].cpu().detach().numpy())
                path_y.append(point[1].cpu().detach().numpy())
            fig = plt.figure()
            plt.plot(path_x, path_y)
            plt.imshow(sdf.cpu().detach().numpy(), origin="lower")
            plt.scatter(*start_pos.cpu().detach().numpy())
            plt.scatter(*end_pos.cpu().detach().numpy())
            experiment.log_figure(fig, step=i)



# %% [markdown]
# # Run Training

# %%

experiment.log_parameters(hyper_params)
with experiment.train():
    train(model, optimizer, device, sdf, loss)
experiment.end()


# %% [markdown]
# # Visualize path

# %%
experiment.end()

# %%



