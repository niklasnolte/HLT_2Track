from posixpath import join
import imageio
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from matplotlib import pyplot as plt
from hlt2trk.utils.config import Locations, format_location, Configuration, dirs
from os.path import join
from tqdm import tqdm

plt.style.use(join(dirs.project_root, 'scripts/plot/paper-dark'))
plt.switch_backend("TkAgg")


def train_torch_model(
    cfg: Configuration,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    assert cfg.model.startswith("nn")

    x_train: torch.Tensor = torch.from_numpy(x_train).float()
    y_train: torch.Tensor = torch.from_numpy(y_train).float()[:, None]
    x_val: torch.Tensor = torch.from_numpy(x_val).float()
    y_val: torch.Tensor = torch.from_numpy(y_val).float()[:, None]

    data = TensorDataset(x_train, y_train)
    loader = DataLoader(data, batch_size=4096, shuffle=False)

    def train(
            model, optimizer, scheduler, filename, loss_fun=F.binary_cross_entropy,
            make_gif=False):
        device = cfg.device
        print(f"training on {device}")

        y_val_ = y_val.to(device)
        if make_gif:
            tmp_file = os.path.splitext(filename)[0]
            tmp_file += "{}.png"
            files = []
            fig, ax = plt.subplots(1, 1, dpi=120, figsize=(16, 9))
            ax.set_xlim(-0.1, 1.1)
            plt.tight_layout()

        model.to(device)
        pbar = tqdm(range(EPOCHS))
        for i in pbar:
            if cfg.model in ["nn-one", "nn-inf"] and cfg.sigma_final is not None:
                model.sigmanet.sigma *= (cfg.sigma_final / cfg.sigma_init)**(1 / EPOCHS)
                model.sigmanet.gamma += (cfg.gamma_final - cfg.gamma_init) / EPOCHS
            # Train
            model.train()
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                weights = (y == 0) * y.mean() + (y == 1)
                loss = loss_fun(output, y, weight=weights)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # Validate
            model.eval()
            with torch.no_grad():
                y_pred = model(x_val.to(device))
                weights = (y_val_ == 0) * y_val_.mean() + (y_val_ == 1)
                loss = loss_fun(y_pred, y_val_, weight=weights)
                pred = y_pred.squeeze().cpu().numpy()
                val = y_val.squeeze().numpy()

            auc = roc_auc_score(val, np.clip(pred, 0., 1.))
            acc = max([balanced_accuracy_score(val, pred > x)
                      for x in np.linspace(0.1, .9, 5)])

            desc = f"epoch {i}, loss: {loss.item():.4f}, auc: {auc:.4f}, acc: {acc:.4f}"
            if cfg.model in ["nn-one", "nn-inf"]:
                desc += f" sigma: {model.sigmanet.sigma.item(): .2f}"
                desc += f" lr: {optimizer.param_groups[0]['lr']:.2e}"
            pbar.set_description(desc)

            if make_gif:
                range_ = None  # (0, 1)
                ax.hist(pred[val == 1], bins=100, alpha=0.5,
                        density=True, label="sig preds", range=range_,)
                ax.hist(pred[val == 0], bins=100, alpha=0.5,
                        density=True, label="bkg preds", range=range_, )
                ax.text(0, 0.965,
                        f"Epoch {i + 1}/{EPOCHS}, loss {loss.item():.3f}, auc {auc:.3f}",
                        transform=ax.transAxes, fontsize=20,)

                f = tmp_file.format(i)
                files.append(f)
                plt.savefig(f)
                fig.canvas.draw()
                # plt.pause(0.0001)
                ax.cla()
                fig.canvas.flush_events()
        # build gif
        if make_gif:
            with imageio.get_writer(filename, mode="I") as writer:
                for fn in files:
                    image = imageio.imread(fn)
                    writer.append_data(image)
                    os.remove(fn)
            plt.close()

    EPOCHS = 40
    LR = 1e-1

    torch.manual_seed(1)
    from hlt2trk.models import get_model

    model = get_model(cfg)

    nparams = sum([x.view(-1).shape[0] for x in model.parameters()])
    print(f"model has {nparams} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.996)

    def weighted_mse_loss(input, target, weight=None):
        if weight is None:
            weight = 1
        else:
            if weight.shape != input.shape or target.shape != input.shape:
                raise ValueError("weight [{weight.shape}], input [{input.shape}],\
                target [{target.shape}] must all have the same shape")
        return (weight * (input - target) ** 2).mean()
    train(
        model,
        optimizer,
        scheduler,
        filename=format_location(Locations.train_distribution_gif, cfg),
        loss_fun=F.binary_cross_entropy  # weighted_mse_loss,
    )

    torch.save(model.state_dict(), format_location(Locations.model, cfg))

    with torch.no_grad():
        preds = model.to(torch.device("cpu"))(x_val)
    auc = roc_auc_score(y_val, np.clip(preds, 0., 1.))
    acc = max(balanced_accuracy_score(y_val, preds > i) for i in np.linspace(0, 1, 100))
    print(f"Final Validation Results: \nroc: {auc:.6f}, acc: {acc:.6f}")
    return model
