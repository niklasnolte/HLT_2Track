import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import re
from os.path import join

import hlt2trk.data.meta_info as meta
from sys import argv

feynman = "feynman" in argv

if feynman:
    with open(join(meta.locations.project_root,
                   'scripts/results/solution_MC_preprocessedExp3_train.txt'),
              'r') as f:
        x = f.readlines()[-1]
    x = " ".join(x.split(" ")[5:])
    x = re.sub('x(\\d)', 'x[:,\\g<1>]', x)
    x = x.replace("sin", "np.sin")
    x = x.replace("cos", "np.cos")
    x = x.replace("tan", "np.tan")
    x = x.replace("asin", "np.asin")
    x = x.replace("acos", "np.acos")
    x = x.replace("atan", "np.atan")
    x = x.replace("exp", "np.exp")
    feynman_fun = eval("lambda x: " + x)

_, _, X, Y_truth = meta.get_data_for_training()

torch.manual_seed(2)

# model_sigma = meta.load_model('sigma')
network = 'regular'  # TODO should be get from config
model = meta.load_model(network)

data = TensorDataset(torch.from_numpy(X).float())
loader = DataLoader(data, batch_size=100000, shuffle=False)

Y: np.ndarray = np.zeros(len(X))
Y_sigma: np.ndarray = np.zeros(len(X))
if feynman:
    Y_feynman: np.ndarray = np.zeros(len(X))

idx = 0
with torch.no_grad():
    for (x,) in loader:
        y: np.ndarray = model(x).numpy().flatten()
        # y_sigma: np.ndarray = model_sigma(x).numpy().flatten()

        Y[idx: idx + len(y)] = y
        # Y_sigma[idx: idx + len(y)] = y_sigma
        if feynman:
            y_feynman: np.ndarray = feynman_fun(x.numpy())
            Y_feynman[idx: idx + len(y)] = y_feynman

        idx += len(y)

print("roc regular {:.6f}".format(roc_auc_score(Y_truth, Y)))
# print("roc sigma   {:.6f}".format(roc_auc_score(Y_truth, Y_sigma)))
if feynman:
    print("roc feynman {:.6f}".format(roc_auc_score(Y_truth, Y_feynman)))

print("acc regular {:.6f}".format(
    max([balanced_accuracy_score(Y_truth, Y > i)
         for i in np.linspace(0, 1, 50)])))
# print("acc sigma   {:.6f}".format(max([balanced_accuracy_score(
#     Y_truth, Y_sigma > i) for i in np.linspace(0, 1, 50)])))
if feynman:
    print("acc feynman {:.6f}".format(max([balanced_accuracy_score(
        Y_truth, Y_feynman > i) for i in np.linspace(0, 1, 50)])))
