import imageio
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import hlt2trk.utils.meta_info as meta
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from matplotlib import pyplot as plt
plt.style.use("seaborn")

# TODO change if else statements to network as imported from config.
print(
    f"training {'sigma' if meta.sigma_net else 'regular'} network in\
    {len(meta.features)} dimensions \
    {'on LHCb sim' if meta.lhcb_sim else 'on standalone sim'}"
)

x_train, y_train, x_val, y_val = meta.get_data_for_training()

print(f"mean label: {y_train.mean()}")
print(f"size of data: {len(x_train)}")

x_train: torch.Tensor = torch.from_numpy(x_train).float()
y_train: torch.Tensor = torch.from_numpy(y_train).float()[:, None]
x_val: torch.Tensor = torch.from_numpy(x_val).float()
y_val: torch.Tensor = torch.from_numpy(y_val).float()[:, None]

data = TensorDataset(x_train, y_train)
loader = DataLoader(data, batch_size=512, shuffle=False)


def train(model, optimizer, scheduler, filename=None,
          loss_fun=F.binary_cross_entropy):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    print(f"training on {device}")

    y_val_ = y_val.to(device)

    if filename is not None:
        tmp_file = os.path.splitext(filename)[0]
        tmp_file += "{}.png"
        files = []
    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(16, 9))
    ax.set_xlim(-0.1, 1.1)
    plt.tight_layout()
    model.to(device)
    for i in range(EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            weights = (y == 0) * y.mean() + (y == 1)
            loss = loss_fun(output, y, weight=weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            y_pred = model(x_val.to(device))
            weights = (y_val_ == 0) * y_val_.mean() + (y_val_ == 1)
            loss = loss_fun(y_pred, y_val_, weight=weights)
            pred = y_pred.squeeze().cpu()
            val = y_val.squeeze()
            range_ = (0, 1)
            ax.hist(
                pred[val == 1].numpy(),
                bins=100,
                alpha=0.5,
                density=True,
                label="sig preds",
                range=range_,
            )
            ax.hist(
                pred[val == 0].numpy(),
                bins=100,
                alpha=0.5,
                density=True,
                label="bkg preds",
                range=range_,
            )
            auc = roc_auc_score(val, pred)
            acc = max([balanced_accuracy_score(val, pred > x)
                       for x in np.linspace(0, 1, 20)])
            print(f"epoch {i}, auc: {auc:.4f}, acc: {acc:.4f}", end="\r")
            ax.text(
                0,
                0.965,
                f"Epoch {i + 1}/{EPOCHS},\
                loss {loss.item():.3f},\
                auc {auc:.3f}",
                transform=ax.transAxes,
                fontsize=20,
            )
        scheduler.step()
        if filename is not None:
            f = tmp_file.format(i)
            files.append(f)
            plt.savefig(f)
        fig.canvas.draw()
        # plt.pause(0.0001)
        ax.cla()
        fig.canvas.flush_events()
    if filename is not None:
        # build gif
        with imageio.get_writer(filename, mode="I") as writer:
            for fn in files:
                image = imageio.imread(fn)
                writer.append_data(image)
                os.remove(fn)
    plt.close()


EPOCHS = 50
LR = 1e-2

torch.manual_seed(2)
from hlt2trk.models import default_model as model

nparams = sum([x.view(-1).shape[0] for x in model.parameters()])
print(
    f"model has {nparams} parameters")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer, gamma=0.99)

train(
    model,
    optimizer,
    scheduler,
    filename=f"plots/train_{meta.path_suffix}.gif",
    loss_fun=F.binary_cross_entropy,
)

torch.save(model.state_dict(), meta.locations.model)

with torch.no_grad():
    preds = model.to(torch.device("cpu"))(x_val)
auc = roc_auc_score(y_val, preds)
acc = max(balanced_accuracy_score(y_val, preds > i)
          for i in np.linspace(0, 1, 100))
print(f"\nroc: {auc:.6f}, acc: {acc:.6f}")
