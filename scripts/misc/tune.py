from hlt2trk.utils.config import Configuration, dirs
from os.path import join, abspath, dirname
from sys import argv

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from hlt2trk.utils.data import get_data_for_training
from monotonenorm import project_norm, direct_norm, GroupSort

DEVICE = "cpu"  # torch.device("cuda:0")
BATCHSIZE = 64
CLASSES = 1
EPOCHS = 50
TIMEOUT = 60 * 60 * .2
TRIALS = 100
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 10
N_VALID_EXAMPLES = BATCHSIZE * 10
NORMALIZE = True
SAVE_PATH = join(dirs.results, 'regular-tuned.txt')
# try:
#     SAVE_PATH = [x.split('=')[1] for x in argv if 'save_to=' in x][0]
# except IndexError:
#     SAVE_PATH = None

try:
    nn_norm = eval([x.split('=')[1] for x in argv if 'norm=' in x][0])
except IndexError:
    def nn_norm(x):
        return x

# nn_norm = infnorm if 'infnorm' in argv else lambda x: x


def define_model(trial: optuna.trial.Trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 2, 32, 2)
    n_groupsort = n_units // trial.suggest_int("n_groupsort", 1, 2)
    layers = []

    in_features = 4
    # for i in range(n_layers):
    #     out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
    #     layers.append(nn_norm(nn.Linear(in_features, out_features)))
    #     layers.append(nn.ReLU())
    #     p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
    #     layers.append(nn.Dropout(p))
    #     in_features = out_features
    for i in range(n_layers):
        out_features = n_units
        layers.append(nn_norm(nn.Linear(in_features, out_features)))
        layers.append(GroupSort(n_groupsort))
        p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn_norm(nn.Linear(in_features, CLASSES)))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def get_loaders():
    # Load Data
    X_train, y_train, X_val, y_val = [
        torch.from_numpy(x).float().view(len(x), -1)
        for x in get_data_for_training(
            Configuration(normalize=NORMALIZE,
                          features=['fdchi2', 'sumpt', 'minipchi2', 'vchi2'],
                          data_type='lhcb',
                          signal_type='heavy-flavor',
                          presel_conf={'svPT': 1000, 'trkPT': 200, 'svchi2': 100,
                                       'ipcuttrain': 10},
                          ))
    ]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=6,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=len(X_val),
        shuffle=False,
        num_workers=6,
    )

    return train_loader, val_loader


def objective(trial: optuna.trial.Trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_loader, val_loader = get_loaders()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(
                data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        with torch.no_grad():
            y_score = []
            y_true = []
            for batch_idx, (data, target) in enumerate(val_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(
                    data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)

                y_score.append(output)
                y_true.append(target)
        y_score = torch.cat(y_score).cpu()
        y_true = torch.cat(y_true).cpu()
        auc = roc_auc_score(y_true, y_score)
        trial.report(auc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return auc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", )
    study.optimize(objective, n_trials=TRIALS, timeout=TIMEOUT)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    out_string = f"Best trial:"
    out_string += f"\nValue: {trial.value}"

    out_string += "\nParams: "
    for key, value in trial.params.items():
        out_string += "\n    {}: {}".format(key, value)

    print(out_string)
    if SAVE_PATH is not None:
        with open(SAVE_PATH, 'w') as f:
            f.write(out_string)
        print(f'saved to: {join(abspath(dirname(SAVE_PATH)), SAVE_PATH)}')
