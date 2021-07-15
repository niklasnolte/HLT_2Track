import imageio
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from matplotlib import pyplot as plt
from hlt2trk.utils.config import Locations, format_location, Configuration

plt.style.use("seaborn")


def train_torch_model(
    cfg: Configuration,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    assert cfg.model in ["regular", "sigma"]

    x_train: torch.Tensor = torch.from_numpy(x_train).float()
    y_train: torch.Tensor = torch.from_numpy(y_train).float()[:, None]
    x_val: torch.Tensor = torch.from_numpy(x_val).float()
    y_val: torch.Tensor = torch.from_numpy(y_val).float()[:, None]

    data = TensorDataset(x_train, y_train)
    loader = DataLoader(data, batch_size=512, shuffle=False)

    def train(model, optimizer, scheduler, filename, loss_fun=F.binary_cross_entropy):
        device = cfg.device
        print(f"training on {device}")

        y_val_ = y_val.to(device)

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
                acc = max(
                    [
                        balanced_accuracy_score(val, pred > x)
                        for x in np.linspace(0, 1, 20)
                    ]
                )
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
            f = tmp_file.format(i)
            files.append(f)
            plt.savefig(f)
            fig.canvas.draw()
            # plt.pause(0.0001)
            ax.cla()
            fig.canvas.flush_events()
        # build gif
        with imageio.get_writer(filename, mode="I") as writer:
            for fn in files:
                image = imageio.imread(fn)
                writer.append_data(image)
                os.remove(fn)
        plt.close()

    EPOCHS = 30
    LR = 1e-2

    torch.manual_seed(2)
    from hlt2trk.models import get_model

    model = get_model(cfg)

    nparams = sum([x.view(-1).shape[0] for x in model.parameters()])
    print(f"model has {nparams} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    train(
        model,
        optimizer,
        scheduler,
        filename=format_location(Locations.train_distribution_gif, cfg),
        loss_fun=F.binary_cross_entropy,
    )

    torch.save(model.state_dict(), format_location(Locations.model, cfg))

    with torch.no_grad():
        preds = model.to(torch.device("cpu"))(x_val)
    auc = roc_auc_score(y_val, preds)
    acc = max(balanced_accuracy_score(y_val, preds > i) for i in np.linspace(0, 1, 100))
    print(f"\nroc: {auc:.6f}, acc: {acc:.6f}")
